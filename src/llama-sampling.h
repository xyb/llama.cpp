#pragma once

#include "llama-grammar.h"

#include <random>
#include <unordered_map>

struct llama_vocab;
struct llama_grammar;

using llama_token_cnt = std::unordered_map<llama_token, int>;

struct llama_sampling {
    llama_sampling(const struct llama_vocab & vocab);
    ~llama_sampling();

    llama_sampling_params params;

    std::string grammar_str;
    std::string grammar_root;

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply

    // state

    std::mt19937 rng;

    const struct llama_vocab & vocab;

    struct llama_grammar * grammar = nullptr;

    ring_buffer<llama_token> prev;

    // mirostat sampler state
    float mirostat_mu;

    mutable int64_t t_total_us = 0;

    mutable int32_t n_sample = 0;
};

//
// internal API
//

struct llama_sampling * llama_sampling_init_impl(const struct llama_vocab & vocab, struct llama_sampling_params params);

void llama_sampling_free_impl(struct llama_sampling * sampling);

struct llama_sampling * llama_sampling_cp_impl(const struct llama_sampling & smpl);

void llama_sampling_reset_impl(struct llama_sampling & smpl);

// TODO: move the API below as member functions of llama_sampling
void llama_sampling_set_rng_seed_impl  (struct llama_sampling & smpl, uint32_t seed);
void llama_sampling_set_grammar_impl   (struct llama_sampling & smpl, const char * grammar_str, const char * grammar_root);
void llama_sampling_set_logit_bias_impl(struct llama_sampling & smpl, int32_t n_logit_bias, const llama_logit_bias * logit_bias);

void llama_sampling_softmax_impl  (struct llama_token_data_array * candidates);
void llama_sampling_top_k_impl    (struct llama_token_data_array * candidates, int32_t k, size_t min_keep);
void llama_sampling_top_p_impl    (struct llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_min_p_impl    (struct llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_tail_free_impl(struct llama_token_data_array * candidates, float z, size_t min_keep);
void llama_sampling_typical_impl  (struct llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_entropy_impl  (struct llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val);
void llama_sampling_temp_impl     (struct llama_token_data_array * candidates, float temp);
void llama_sampling_grammar_impl  (struct llama_token_data_array * candidates, const struct llama_grammar & grammar);

void llama_sampling_penalties_impl(
       llama_token_data_array * candidates,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

void llama_sampling_cfg_impl(
        struct llama_sampling & smpl,
                        float * logits,
                        float * logits_guidance);

/// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
llama_token llama_sampling_sample_mirostat_impl   (struct llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, int32_t m, int32_t n_vocab, float & mu);

/// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
llama_token llama_sampling_sample_mirostat_v2_impl(struct llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, float & mu);

llama_token llama_sampling_sample_greedy_impl     (struct llama_token_data_array * candidates);
llama_token llama_sampling_sample_impl            (struct llama_token_data_array * candidates, std::mt19937 & rng);

void llama_sampling_accept_impl(struct llama_sampling & smpl, llama_token token, bool apply_grammar);

llama_token llama_sampling_prev_impl(const struct llama_sampling & smpl, int ith);
int         llama_sampling_n_prev_impl(const struct llama_sampling & smpl);
