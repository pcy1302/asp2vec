import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import pickle as pkl
from embedder import embedder
import glob

class asp2vec(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.clip_max = torch.FloatTensor([1.0]).to(self.device)

    def train_DW(self):
        # Train Deepwalk for warm-up
        model_DW = modeler_DW(self.args).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model_DW.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)

        f_list = glob.glob("{}/batch*".format(self.batch_path))
        num_batches = len(f_list)
        if num_batches == 0:
            num_batches = self.generate_training_batch()

        # Start training
        print("[{}] Start warm-up".format(currentTime()))
        for epoch in range(0, self.iter_max):
            self.batch_loss = 0
            for batch_idx in range(num_batches):
                batch_f_name = "{}/batch_{}.pkl".format(self.batch_path, batch_idx)
                batch = pkl.load(open(batch_f_name, "rb"))
                pairs, negs, _, _ = batch
                pairs, negs = torch.LongTensor(pairs), torch.LongTensor(negs)
                pairs, negs = pairs.to(self.device), negs.to(self.device)
                optimizer.zero_grad()
                loss = model_DW(pairs, negs)
                self.batch_loss += loss.item()
                loss.backward()
                optimizer.step()

            model_DW.center_embedding.weight.data.div_(torch.max(torch.norm(model_DW.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(model_DW.center_embedding.weight.data))
            model_DW.context_embedding.weight.data.div_(torch.max(torch.norm(model_DW.context_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(model_DW.context_embedding.weight.data))

            if epoch % self.eval_freq == 0:
                center_emb_intermediate, context_emb_intermediate = model_DW.get_embeds()
                converged = self.evaluate_DW(epoch, center_emb_intermediate, context_emb_intermediate)

                self.saved_model_DW.append([context_emb_intermediate, center_emb_intermediate])

                if converged:
                    self.print_result(epoch=epoch, warmup=True, isFinal='current')
                    print("Warm-up converged on epoch {}!".format(epoch))

                    # save best model
                    idx = self.print_result(warmup=True, isFinal='Final')
                    is_converged = True
                    break
                else:
                    self.print_result(epoch=epoch, warmup=True, isFinal='current')
                    is_converged = False


        if not is_converged:
            # Final evaluation
            center_emb_final, context_emb_final = model_DW.get_embeds()
            self.evaluate_DW('Final', center_emb_final, context_emb_final)

            # return best model
            idx = self.print_result(warmup=True, isFinal='Final')

        embeds = self.saved_model_DW[idx]
        context_embed, center_embed = embeds[0], embeds[1]
        embed = (context_embed + center_embed) / 2.0

        return embed


    def evaluate_DW(self, epoch, center_emb, context_emb):
        avg_emb = (context_emb + center_emb) / 2.0
        with torch.no_grad():
            self.eval_link_prediction(avg_emb)

        return self.is_converged(epoch)

    def training(self):
        if self.isInit:
            pretrained_embed = self.train_DW()
            self.early_stop = 0
            self.result_dict = {}
        else:
            pretrained_embed = None
        model_asp2vec = modeler_asp2vec(self.args, pretrained_embed).to(self.device)
        parameters = filter(lambda p: p.requires_grad, model_asp2vec.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)
        f_list = glob.glob("{}/batch*".format(self.batch_path))
        num_batches = len(f_list)
        if num_batches == 0:
            num_batches = self.generate_training_batch()

        # Start training
        print("[{}] Start training asp2vec".format(currentTime()))
        for epoch in range(0, self.iter_max):
            self.batch_loss = 0
            if self.isReg:
                self.total_div_reg = 0

            for batch_idx in range(num_batches):
                batch_f_name = "{}/batch_{}.pkl".format(self.batch_path, batch_idx)
                batch = pkl.load(open(batch_f_name, "rb"))
                pairs, negs, offsets, lists = batch
                pairs, negs, offsets, lists = torch.LongTensor(pairs), torch.LongTensor(negs), torch.LongTensor(offsets), torch.LongTensor(lists)
                pairs, negs, offsets, lists = pairs.to(self.device), negs.to(self.device), offsets.to(self.device), lists.to(self.device)
                optimizer.zero_grad()
                if not self.isReg:
                    loss = model_asp2vec(batch_idx, pairs, negs, offsets, lists)
                else:
                    loss, div_reg = model_asp2vec(batch_idx, pairs, negs, offsets, lists)
                    self.total_div_reg += div_reg
                self.batch_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % self.eval_freq == 0:
                center_emb_intermediate, aspect_emb_intermediate = model_asp2vec.get_embeds()
                converged = self.evaluate_asp2vec(epoch, center_emb_intermediate, aspect_emb_intermediate)
                self.saved_model_asp2vec.append([center_emb_intermediate, aspect_emb_intermediate])

                if converged:
                    self.print_result(epoch=epoch, isFinal='current')
                    print("Converged on epoch {}!\n".format(epoch))
                    self.print_result(isFinal='Final')
                    exit(0)
                else:
                    self.print_result(epoch=epoch, isFinal='current')

        # Final evaluation
        center_emb_final, aspect_emb_final = model_asp2vec.get_embeds()
        self.evaluate_asp2vec('Final', center_emb_final, aspect_emb_final)
        self.print_result(isFinal='Final')

    def evaluate_asp2vec(self, epoch, center_emb, aspect_emb):
        aspect_emb_reshaped = aspect_emb.view(self.num_aspects, self.num_nodes, self.dim).permute(1, 0, 2).contiguous()
        aspect_emb_avg = torch.mean(aspect_emb_reshaped, 1)
        total_emb_avg = (aspect_emb_avg + center_emb) / 2.0

        with torch.no_grad():
            self.eval_link_prediction(total_emb_avg)

        return self.is_converged(epoch)

class modeler_DW(nn.Module):
    def __init__(self, args):
        super(modeler_DW, self).__init__()
        self.num_nodes = args.num_nodes
        self.logsigmoid = nn.LogSigmoid()
        self.dim = args.dim
        self.device = args.device
        self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
        self.context_embedding = nn.Embedding(args.num_nodes, args.dim)
        self.clip_max = torch.FloatTensor([1.0])
        self.init_weights()


    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.context_embedding.weight, mean=0.0, std=0.01)
        self.center_embedding.weight.data.div_(torch.max(torch.norm(self.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(self.center_embedding.weight.data))
        self.context_embedding.weight.data.div_(torch.max(torch.norm(self.context_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(self.context_embedding.weight.data))

    def forward(self, pairs, negs):
        centers = pairs[:, 0]
        contexts = pairs[:, 1]
        embed_contexts = self.context_embedding(contexts)
        embed_contexts_neg = self.context_embedding(negs)

        total_centers_idxs = torch.cat([k * self.num_nodes + centers.unsqueeze(1) for k in range(1)],1)
        embed_centers = self.center_embedding(total_centers_idxs)

        score_pos = torch.bmm(embed_centers, embed_contexts.unsqueeze(-1)).squeeze(-1)
        score_pos = -F.logsigmoid(score_pos)

        score_neg = torch.bmm(embed_centers, embed_contexts_neg.permute(0, 2, 1))
        score_neg = -torch.sum(F.logsigmoid(-score_neg), dim=2)

        sg_loss = score_pos + score_neg
        sg_loss = torch.mean(sg_loss)

        final_loss = sg_loss

        return final_loss


    def get_embeds(self):
        with torch.no_grad():
            embed_centers = self.center_embedding.weight.data
            embed_contexts = self.context_embedding.weight.data
            embed_centers = embed_centers.cpu()
            embed_contexts = embed_contexts.cpu()

            return embed_centers, embed_contexts

class modeler_asp2vec(nn.Module):
    def __init__(self, args, pretrained_embed=None):
        super(modeler_asp2vec, self).__init__()
        self.num_aspects = args.num_aspects
        self.num_nodes = args.num_nodes
        self.logsigmoid = nn.LogSigmoid()
        self.dim = args.dim
        self.device = args.device
        self.aspect_embedding = nn.Embedding(args.num_nodes * args.num_aspects, args.dim)
        self.center_embedding = nn.Embedding(args.num_nodes, args.dim)
        self.pooling = args.pooling
        self.isInit = args.isInit

        self.isReg = args.isReg
        if self.isReg:
            self.reg_coef = args.reg_coef
            self.threshold = args.threshold

        self.isSoftmax = args.isSoftmax
        if self.isSoftmax:
            self.isGumbelSoftmax = args.isGumbelSoftmax
            self.isNormalSoftmax = args.isNormalSoftmax
            if self.isGumbelSoftmax:
                self.tau_gumbel = args.tau_gumbel
                self.isHard = args.isHard

        if self.isInit:
            self.init_weights(pretrained_embed=pretrained_embed)
        else:
            self.init_weights()

    def init_weights(self, pretrained_embed=None):
        if pretrained_embed is not None:
            # Initialize embeddings
            self.center_embedding.weight = torch.nn.Parameter(pretrained_embed)
            with torch.no_grad():
                for k in range(self.num_aspects):
                    self.aspect_embedding.weight[k * self.num_nodes: (k + 1) * self.num_nodes] = torch.nn.Parameter(pretrained_embed)
        else:
            nn.init.normal_(self.aspect_embedding.weight.data, mean=0.0, std=0.01)
            nn.init.normal_(self.center_embedding.weight.data, mean=0.0, std=0.01)

    def forward(self, batch_idx, pairs, negs, offsets, lists):
        centers = pairs[:, 0]
        embed_centers = self.center_embedding(centers)  # N x dim
        embed_contexts_means = torch.stack([F.embedding_bag(lists, self.aspect_embedding.weight[k * self.num_nodes: (k + 1) * self.num_nodes], offsets, mode=self.pooling) for k in range(self.num_aspects)], 1)  # N x K x dim

        if self.isSoftmax:
            # Apply softmax
            aspect_softmax = torch.bmm(embed_contexts_means, embed_centers.unsqueeze(-1)).squeeze(-1)  # N x K
            # 1-1. Gumbel Softmax
            if self.isGumbelSoftmax:
                # In fact, following the original Gumbel-softmax, the input for F.gumbel_softmax() should be logit (i.e., unnormalized log probabilities.) 
                # However, we found that unnormalized probabilities without log are numerically more stable, and performs on par with logit.
                aspect_softmax = F.gumbel_softmax(aspect_softmax, tau=self.tau_gumbel, hard=self.isHard)
            elif self.isNormalSoftmax:
                # 1-2. Softmax
                aspect_softmax = F.softmax(aspect_softmax, dim=1)

        contexts = pairs[:, 1]
        total_contexts_idxs = torch.cat([k * self.num_nodes + contexts.unsqueeze(1) for k in range(self.num_aspects)],
                                        1)
        aspect_embedding_context = self.aspect_embedding(total_contexts_idxs)  # N x K x dim

        score_pos = torch.bmm(aspect_embedding_context, embed_centers.unsqueeze(-1)).squeeze(-1)  # (N x K x dim) x (N x dim x 1) = (N x K)
        score_pos = -F.logsigmoid(score_pos)

        embed_contexts_negs = [self.aspect_embedding(k * self.num_nodes + negs) for k in range(self.num_aspects)]  # [N x num_neg x dim] * K
        score_negs = [torch.bmm(embed_contexts_neg, embed_centers.unsqueeze(-1)).squeeze(-1) for k, embed_contexts_neg in enumerate(embed_contexts_negs)]
        score_neg = torch.stack([-torch.sum(F.logsigmoid(-score_neg), dim=1) for score_neg in score_negs], 1)

        if self.isSoftmax:
            sg_loss = aspect_softmax * (score_pos + score_neg)
        else:
            sg_loss = (score_pos + score_neg) / self.num_aspects

        sg_loss = torch.mean(sg_loss)

        final_loss = sg_loss

        # Aspect regularization
        if self.isReg:
            div_metric = None
            # N x K x dim
            aspect_emb_reshaped = self.aspect_embedding.weight.view(self.num_aspects, self.num_nodes, self.dim).permute(1, 0, 2).contiguous()
            for i in range(self.num_aspects):
                for j in range(i + 1, self.num_aspects):
                    sim_matrix = F.cosine_similarity(aspect_emb_reshaped[:, i, :], aspect_emb_reshaped[:, j, :])
                    mask = torch.abs(sim_matrix) > self.threshold
                    if i == 0 and j == 1:
                        div_metric = (torch.abs(torch.masked_select(sim_matrix, mask))).sum()
                    else:
                        div_metric += (torch.abs(torch.masked_select(sim_matrix, mask))).sum()

            div_reg = self.reg_coef * div_metric

            final_loss += div_reg
            return final_loss, (self.reg_coef * div_metric).item()

        return final_loss

    def get_embeds(self):
        with torch.no_grad():
            center_embedding = self.center_embedding.weight.data
            aspect_embedding = self.aspect_embedding.weight.data

            center_embedding = center_embedding.cpu()
            aspect_embedding = aspect_embedding.cpu()

            return center_embedding, aspect_embedding


