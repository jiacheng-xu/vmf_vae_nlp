class Beam(object):
    def __init__(self, opt, tokens, log_probs, state, prev_attn, p_gens, coverage=None, three_grams=[], bi_grams=[]):
        """    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.opt = opt
        self.tokens = tokens
        self.coverage = coverage
        self.log_probs = log_probs
        self.state = state
        self.prev_attn = prev_attn
        self.p_gens = p_gens
        self.three_grams = three_grams
        self.bi_grams = bi_grams

    def extend(self, opt, token, log_prob, state, prev_attn, coverage=None, bi_gram=None, three_gram=None, p_gen=None):
        if three_gram is None:
            return Beam(opt=opt, tokens=self.tokens + [token],
                        log_probs=self.log_probs + [log_prob],
                        state=state, prev_attn=self.prev_attn + [prev_attn], p_gens=self.p_gens + [p_gen],
                        coverage=coverage,
                        three_grams=self.three_grams, bi_grams=self.bi_grams)
        else:
            if opt.avoid:
                if self.avid_repeatition(3, three_gram):
                    log_prob -= 10
                if self.avid_repeatition(2, bi_gram):
                    log_prob -= 3
            self.three_grams.append(three_gram)
            self.bi_grams.append(bi_gram)
            new_three_gram = self.three_grams
            new_bi_gram = self.bi_grams
            return Beam(opt=opt, tokens=self.tokens + [token],
                        log_probs=self.log_probs + [log_prob],
                        state=state, prev_attn=self.prev_attn + [prev_attn], p_gens=self.p_gens + [p_gen],
                        coverage=coverage, three_grams=new_three_gram, bi_grams=new_bi_gram)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def latest_attn(self):
        return self.prev_attn[-1]

    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob() / (len(self.tokens) - 1)

    def avid_repeatition(self, ngram, candidate_gram):
        # If the latest three grams appear previously, return True
        if len(self.tokens) > ngram:
            # latest_3 = "%d_%d_%d" % (self.tokens[-3], self.tokens[-2], self.tokens[-1])
            if ngram == 3:
                if candidate_gram in self.three_grams:
                    return True
            elif ngram == 2:
                if candidate_gram in self.bi_grams:
                    return True
        return False

    @staticmethod
    def sort_hyps(hyps):
        """Return a list of Hypothesis objects, sorted by descending average log probability"""
        return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)
