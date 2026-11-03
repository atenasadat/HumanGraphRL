"""JSON token constraints through LM Format Enforcer's public core API.

The 0.11.3 Transformers adapter imports an obsolete tokenizer symbol under
Transformers 5. Use TokenEnforcer directly, without monkey-patching Transformers.
Tokenizer preprocessing follows the upstream integration's word-boundary rule:
https://github.com/noamgat/lm-format-enforcer
"""
from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def json_prefix_filter(tokenizer, schema):
    """Create fresh per-generation parser state; reuse only tokenizer metadata."""
    data = getattr(tokenizer, '_paper_json_tokenizer_data', None)
    if data is None:
        anchor = tokenizer.encode('0', add_special_tokens=False)[-1]
        special = set(tokenizer.all_special_ids)
        regular = []
        for token_id in range(len(tokenizer)):
            if token_id in special:
                continue
            contextual = tokenizer.decode([anchor, token_id], clean_up_tokenization_spaces=False)[1:]
            standalone = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            regular.append((token_id, contextual, len(contextual) > len(standalone)))
        if tokenizer.eos_token_id is None:
            raise ValueError('Constrained judge requires a tokenizer EOS token')
        def decode(ids):
            return tokenizer.decode(ids, clean_up_tokenization_spaces=False).rstrip('\ufffd')
        data = TokenEnforcerTokenizerData(regular, decode, tokenizer.eos_token_id, False, len(tokenizer))
        tokenizer._paper_json_tokenizer_data = data
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))

    def allowed_tokens(batch_id, input_ids):
        return enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens
    return allowed_tokens
