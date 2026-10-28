import collections
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize


class PartialFC(torch.nn.Module):
    """
    Partial Fully Connected layer for distributed training.

    This class implements a partial fully connected layer that can be used in distributed training.
    It supports sampling a subset of classes for each forward pass, which can be useful for large-scale classification tasks.
    """
    _version = 1 

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Initialize the PartialFC layer.

        Args:
            margin_loss (Callable): Margin loss function.
            embedding_size (int): Size of the embedding.
            num_classes (int): Total number of classes.
            sample_rate (float, optional): Sampling rate for classes. Defaults to 1.0.
            fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        """
        super(PartialFC, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.weight: torch.Tensor
        self.weight_mom: torch.Tensor
        self.weight_activated: torch.nn.Parameter
        self.weight_activated_mom: torch.Tensor
        self.is_updated: bool = True
        self.init_weight_update: bool = True

        if self.sample_rate < 1:
            self.register_buffer("weight",
                tensor=torch.normal(0, 0.01, (self.num_local, embedding_size)))
            self.register_buffer("weight_mom",
                tensor=torch.zeros_like(self.weight))
            self.register_parameter("weight_activated",
                param=torch.nn.Parameter(torch.empty(0, 0)))
            self.register_buffer("weight_activated_mom",
                tensor=torch.empty(0, 0))
            self.register_buffer("weight_index",
                tensor=torch.empty(0, 0))
        else:
            self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise ValueError("margin_loss must be a callable")

    @torch.no_grad()
    def sample(self, 
        labels: torch.Tensor, 
        index_positive: torch.Tensor, 
        optimizer: torch.optim.Optimizer):
        """
        Sample a subset of classes for the current forward pass.

        Args:
            labels (torch.Tensor): Labels of the current batch.
            index_positive (torch.Tensor): Indices of positive samples.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
        """
        positive = torch.unique(labels[index_positive], sorted=True).cuda()
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_local]).cuda()
            perm[positive] = 2.0
            index = torch.topk(perm, k=self.num_sample)[1].cuda()
            index = index.sort()[0].cuda()
        else:
            index = positive
        self.weight_index = index

        labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        
        self.weight_activated = torch.nn.Parameter(self.weight[self.weight_index])
        self.weight_activated_mom = self.weight_mom[self.weight_index]
        
        if isinstance(optimizer, torch.optim.SGD):
            optimizer.state.pop(optimizer.param_groups[-1]["params"][0], None)
            optimizer.param_groups[-1]["params"][0] = self.weight_activated
            optimizer.state[self.weight_activated]["momentum_buffer"] = self.weight_activated_mom
        else:
            raise ValueError("Optimizer must be SGD")

    @torch.no_grad()
    def update(self):
        """
        Update the global weights with the partial weights.
        """
        if self.init_weight_update:
            self.init_weight_update = False
            return

        if self.sample_rate < 1:
            self.weight[self.weight_index] = self.weight_activated
            self.weight_mom[self.weight_index] = self.weight_activated_mom

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Forward pass of the PartialFC layer.

        Args:
            local_embeddings (torch.Tensor): Embeddings of the current batch.
            local_labels (torch.Tensor): Labels of the current batch.
            optimizer (torch.optim.Optimizer): Optimizer used for training.

        Returns:
            torch.Tensor: Loss value.
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()
        self.update()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            "last batch size do not equal current batch size: {} vs {}".format(
            self.last_batch_size, batch_size))

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            self.sample(labels, index_positive, optimizer)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Get the state dictionary of the PartialFC layer.

        Args:
            destination (collections.OrderedDict, optional): Destination dictionary. Defaults to None.
            prefix (str, optional): Prefix for the state dictionary. Defaults to "".
            keep_vars (bool, optional): Whether to keep variables. Defaults to False.

        Returns:
            collections.OrderedDict: State dictionary.
        """
        if destination is None: 
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        if self.sample_rate < 1:
            destination["weight"] = self.weight.detach()
        else:
            destination["weight"] = self.weight_activated.data.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load the state dictionary into the PartialFC layer.

        Args:
            state_dict (collections.OrderedDict): State dictionary.
            strict (bool, optional): Whether to strictly enforce the state dictionary. Defaults to True.
        """
        if self.sample_rate < 1:
            self.weight = state_dict["weight"].to(self.weight.device)
            self.weight_mom.zero_()
            self.weight_activated.data.zero_()
            self.weight_activated_mom.zero_()
            self.weight_index.zero_()
        else:
            self.weight_activated.data = state_dict["weight"].to(self.weight_activated.data.device)


class PartialFCAdamW(torch.nn.Module):
    """
    Partial Fully Connected layer for distributed training with AdamW optimizer.

    This class implements a partial fully connected layer that can be used in distributed training.
    It supports sampling a subset of classes for each forward pass, which can be useful for large-scale classification tasks.
    """
    def __init__(self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,):
        """
        Initialize the PartialFCAdamW layer.

        Args:
            margin_loss (Callable): Margin loss function.
            embedding_size (int): Size of the embedding.
            num_classes (int): Total number of classes.
            sample_rate (float, optional): Sampling rate for classes. Defaults to 1.0.
            fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        """
        super(PartialFCAdamW, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.weight: torch.Tensor
        self.weight_exp_avg: torch.Tensor
        self.weight_exp_avg_sq: torch.Tensor
        self.weight_activated: torch.nn.Parameter
        self.weight_activated_exp_avg: torch.Tensor
        self.weight_activated_exp_avg_sq: torch.Tensor

        self.is_updated: bool = True
        self.init_weight_update: bool = True

        if self.sample_rate < 1:
            self.register_buffer("weight",
                tensor=torch.normal(0, 0.01, (self.num_local, embedding_size)))
            self.register_buffer("weight_exp_avg",
                tensor=torch.zeros_like(self.weight))
            self.register_buffer("weight_exp_avg_sq",
                tensor=torch.zeros_like(self.weight))
            self.register_parameter("weight_activated",
                param=torch.nn.Parameter(torch.empty(0, 0)))
            self.register_buffer("weight_activated_exp_avg",
                tensor=torch.empty(0, 0))
            self.register_buffer("weight_activated_exp_avg_sq",
                tensor=torch.empty(0, 0))
        else:
            self.weight_activated = torch.nn.Parameter(
                torch.normal(0, 0.01, (self.num_local, embedding_size))
            )
        self.step = 0

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise ValueError("margin_loss must be a callable")

    @torch.no_grad()
    def sample(self, labels, index_positive, optimizer):
        """
        Sample a subset of classes for the current forward pass.

        Args:
            labels (torch.Tensor): Labels of the current batch.
            index_positive (torch.Tensor): Indices of positive samples.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
        """
        self.step += 1
        positive = torch.unique(labels[index_positive], sorted=True).cuda()
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_local]).cuda()
            perm[positive] = 2.0
            index = torch.topk(perm, k=self.num_sample)[1].cuda()
            index = index.sort()[0].cuda()
        else:
            index = positive
        self.weight_index = index
        labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        self.weight_activated = torch.nn.Parameter(self.weight[self.weight_index])
        self.weight_activated_exp_avg = self.weight_exp_avg[self.weight_index]
        self.weight_activated_exp_avg_sq = self.weight_exp_avg_sq[self.weight_index]

        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            optimizer.state.pop(optimizer.param_groups[-1]["params"][0], None)
            optimizer.param_groups[-1]["params"][0] = self.weight_activated
            optimizer.state[self.weight_activated]["exp_avg"] = self.weight_activated_exp_avg
            optimizer.state[self.weight_activated]["exp_avg_sq"] = self.weight_activated_exp_avg_sq
            optimizer.state[self.weight_activated]["step"] = self.step
        else:
            raise ValueError("Optimizer must be Adam or AdamW")

    @torch.no_grad()
    def update(self):
        """
        Update the global weights with the partial weights.
        """
        if self.init_weight_update:
            self.init_weight_update = False
            return

        if self.sample_rate < 1:
            self.weight[self.weight_index] = self.weight_activated
            self.weight_exp_avg[self.weight_index] = self.weight_activated_exp_avg
            self.weight_exp_avg_sq[self.weight_index] = self.weight_activated_exp_avg_sq

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Forward pass of the PartialFCAdamW layer.

        Args:
            local_embeddings (torch.Tensor): Embeddings of the current batch.
            local_labels (torch.Tensor): Labels of the current batch.
            optimizer (torch.optim.Optimizer): Optimizer used for training.

        Returns:
            torch.Tensor: Loss value.
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()
        self.update()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            "last batch size do not equal current batch size: {} vs {}".format(
            self.last_batch_size, batch_size))

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            self.sample(labels, index_positive, optimizer)

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Get the state dictionary of the PartialFCAdamW layer.

        Args:
            destination (collections.OrderedDict, optional): Destination dictionary. Defaults to None.
            prefix (str, optional): Prefix for the state dictionary. Defaults to "".
            keep_vars (bool, optional): Whether to keep variables. Defaults to False.

        Returns:
            collections.OrderedDict: State dictionary.
        """
        if destination is None: 
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        if self.sample_rate < 1:
            destination["weight"] = self.weight.detach()
        else:
            destination["weight"] = self.weight_activated.data.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load the state dictionary into the PartialFCAdamW layer.

        Args:
            state_dict (collections.OrderedDict): State dictionary.
            strict (bool, optional): Whether to strictly enforce the state dictionary. Defaults to True.
        """
        if self.sample_rate < 1:
            self.weight = state_dict["weight"].to(self.weight.device)
            self.weight_exp_avg.zero_()
            self.weight_exp_avg_sq.zero_()
            self.weight_activated.data.zero_()
            self.weight_activated_exp_avg.zero_()
            self.weight_activated_exp_avg_sq.zero_()
        else:
            self.weight_activated.data = state_dict["weight"].to(self.weight_activated.data.device)


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    Distributed Cross Entropy function.

    This function computes the cross entropy loss in a distributed manner.
    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """
        Forward pass of the distributed cross entropy function.

        Args:
            ctx: Context for the autograd function.
            logits (torch.Tensor): Logits of the model.
            label (torch.Tensor): Labels of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        batch_size = logits.size(0)
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Backward pass of the distributed cross entropy function.

        Args:
            ctx: Context for the autograd function.
            loss_gradient (torch.Tensor): Gradient of the loss.

        Returns:
            torch.Tensor: Gradient of the logits.
        """
        index, logits, label = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    """
    Distributed Cross Entropy module.

    This module wraps the distributed cross entropy function.
    """
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        """
        Forward pass of the distributed cross entropy module.

        Args:
            logit_part (torch.Tensor): Logits of the model.
            label_part (torch.Tensor): Labels of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """
    AllGather function with gradient backward.

    This function gathers tensors from all processes in a distributed manner.
    """
    @staticmethod
    def forward(ctx, tensor, *gather_list):
        """
        Forward pass of the AllGather function.

        Args:
            ctx: Context for the autograd function.
            tensor (torch.Tensor): Tensor to be gathered.
            *gather_list: List of tensors to be gathered.

        Returns:
            tuple: Gathered tensors.
        """
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        """
        Backward pass of the AllGather function.

        Args:
            ctx: Context for the autograd function.
            *grads: Gradients of the gathered tensors.

        Returns:
            tuple: Gradients of the input tensor.
        """
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply