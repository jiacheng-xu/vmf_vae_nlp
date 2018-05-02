def forward_decode_fly(self, emb, lat_code):
    seq_len, batch_sz, _ = emb.size()
    outputs_prob = GVar(torch.FloatTensor(seq_len, batch_sz, self.ntoken))
    outputs = torch.LongTensor(seq_len, batch_sz)

    sos = GVar(torch.ones(batch_sz).long())  # sos id=1
    unk = GVar(torch.ones(batch_sz).long()) * 2  # unk id=2

    emb_t = self.drop(self.encoder(unk)).unsqueeze(0)
    emb_0 = self.drop(self.encoder(sos)).unsqueeze(0)

    if self.input_z:
        # emb_t_comb = torch.cat([emb_t, lat_to_cat], dim=2)
        # emt_0_comb = torch.cat([emb_0, lat_to_cat], dim=2)
        pass



def forward_decode(self, args, input, ntokens):
    """

    :param args:
    :param input: LongTensor [seq_len, batch_sz]
    :param ntokens:
    :return:
        outputs_prob:   Var         seq_len, batch_sz, ntokens
        outputs:        LongTensor  seq_len, batch_sz
        mu, logvar
    """
    seq_len = input.size()[0]
    batch_sz = input.size()[1]

    emb, lat, mu, logvar = self.blstm_enc(input)
    # emb: seq_len, batchsz, hid_dim
    # hidden: ([2(nlayers),10(batchsz),200],[])

    outputs_prob = GVar(torch.FloatTensor(seq_len, batch_sz, ntokens))
    if args.cuda:
        outputs_prob = outputs_prob.cuda()

    outputs = torch.LongTensor(seq_len, batch_sz)

    # First time step sos
    sos = GVar(torch.ones(batch_sz).long())
    unk = GVar(torch.ones(batch_sz).long()) * 2
    if args.cuda:
        sos = sos.cuda()
        unk = unk.cuda()

    lat_to_cat = lat[0][0].unsqueeze(0)
    emb_t = self.drop(self.encoder(unk)).unsqueeze(0)
    emb_0 = self.drop(self.encoder(sos)).unsqueeze(0)
    emb_t_comb = torch.cat([emb_t, lat_to_cat], dim=2)
    emt_0_comb = torch.cat([emb_0, lat_to_cat], dim=2)

    hidden = None

    for t in range(seq_len):
        # input (seq_len, batch, input_size)

        if t == 0:
            emb = emt_0_comb
        else:
            emb = emb_t_comb
        # print(emb.size())

        if hidden is None:
            output, hidden = self.rnn(emb, None)
        else:
            output, hidden = self.rnn(emb, hidden)

        output_prob = self.decoder(self.drop(output))
        output_prob = output_prob.squeeze(0)
        outputs_prob[t] = output_prob
        value, ind = torch.topk(output_prob, 1, dim=1)
        outputs[t] = ind.squeeze(1).data

    return outputs_prob, outputs, mu, logvar


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = GVar(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
