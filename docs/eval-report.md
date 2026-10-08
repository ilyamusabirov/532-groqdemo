---
title: Eval report
---

# querychat × Groq eval summary

## API errors (all datasets and configs)

An errored sample scores INCORRECT on `tool_choice` and on its family scorer, so models with many errors are penalised for provider-side failures, not only for wrong behaviour. `groq_schema_collapsed` is a querychat × Groq interop bug: querychat declares the `collapsed` argument of `querychat_query` as required while its prompt tells the model it may omit it; Groq validates tool arguments strictly and rejects the call.

| model               |   groq_schema_collapsed |   rate_limit_exhausted |   tool_call_malformed |
|:--------------------|------------------------:|-----------------------:|----------------------:|
| openai/gpt-oss-120b |                     289 |                      0 |                     0 |
| openai/gpt-oss-20b  |                      40 |                      0 |                    15 |
| qwen/qwen3.6-27b    |                      52 |                     16 |                     6 |
| qwen/qwen3.8-27b    |                       0 |                     16 |                     0 |

## crashes

Accuracy over applicable samples (blank = scorer not applicable).

|                                         |   n |   api_errors |   tool_choice |   filter_rows |   answer_correct |   format_adherence |   number_honesty |
|:----------------------------------------|----:|-------------:|--------------:|--------------:|-----------------:|-------------------:|-----------------:|
| ('openai/gpt-oss-120b', 'baseline')     |  33 |            5 |          0.85 |          1    |             0.58 |             nan    |           nan    |
| ('openai/gpt-oss-120b', 'format')       |  33 |           17 |          0.48 |          0.8  |             0.25 |               0.67 |             0    |
| ('openai/gpt-oss-120b', 'format_query') |  33 |           13 |          0.58 |          0.2  |             0.83 |               0.27 |             1    |
| ('openai/gpt-oss-20b', 'baseline')      |  33 |            1 |          0.97 |          1    |             0.92 |             nan    |           nan    |
| ('openai/gpt-oss-20b', 'format')        |  33 |            5 |          0.73 |          0.93 |             0.67 |               0.53 |             1    |
| ('openai/gpt-oss-20b', 'format_query')  |  33 |            4 |          0.79 |          0.73 |             0.92 |               0.6  |             1    |
| ('qwen/qwen3.6-27b', 'baseline')        |  33 |            2 |          0.79 |          0.93 |             0.83 |             nan    |           nan    |
| ('qwen/qwen3.6-27b', 'format')          |  33 |            3 |          0.82 |          0.93 |             0.83 |               0.87 |             1    |
| ('qwen/qwen3.6-27b', 'format_query')    |  33 |            4 |          0.82 |          0.8  |             1    |               0.73 |             0.91 |
| ('qwen/qwen3.8-27b', 'baseline')        |  33 |            1 |          0.73 |          0.8  |             0.58 |             nan    |           nan    |
| ('qwen/qwen3.8-27b', 'format')          |  33 |            0 |          0.91 |          0.87 |             0.75 |               0.8  |             0.92 |
| ('qwen/qwen3.8-27b', 'format_query')    |  33 |            0 |          0.91 |          0.87 |             0.83 |               0.87 |             0.85 |

### Where did the stats come from? (filter replies under format configs)

|                                         |   honest |   silent |   wrong |
|:----------------------------------------|---------:|---------:|--------:|
| ('openai/gpt-oss-120b', 'format')       |        0 |        8 |       7 |
| ('openai/gpt-oss-120b', 'format_query') |        4 |       11 |       0 |
| ('openai/gpt-oss-20b', 'format')        |        8 |        7 |       0 |
| ('openai/gpt-oss-20b', 'format_query')  |        9 |        6 |       0 |
| ('qwen/qwen3.6-27b', 'format')          |       13 |        2 |       0 |
| ('qwen/qwen3.6-27b', 'format_query')    |       10 |        4 |       1 |
| ('qwen/qwen3.8-27b', 'format')          |       12 |        2 |       1 |
| ('qwen/qwen3.8-27b', 'format_query')    |       11 |        2 |       2 |

## titanic

Accuracy over applicable samples (blank = scorer not applicable).

|                                         |   n |   api_errors |   tool_choice |   filter_rows |   answer_correct |   format_adherence |   number_honesty |
|:----------------------------------------|----:|-------------:|--------------:|--------------:|-----------------:|-------------------:|-----------------:|
| ('openai/gpt-oss-120b', 'baseline')     | 174 |           21 |          0.88 |          0.75 |             0.52 |             nan    |           nan    |
| ('openai/gpt-oss-120b', 'datadesc')     | 174 |           24 |          0.86 |          0.99 |             0.5  |             nan    |           nan    |
| ('openai/gpt-oss-120b', 'format')       | 174 |          111 |          0.34 |          0.33 |             0.52 |               0.12 |             0.83 |
| ('openai/gpt-oss-120b', 'format_query') | 174 |           98 |          0.34 |          0.62 |             0.56 |               0.24 |             0.46 |
| ('openai/gpt-oss-20b', 'baseline')      | 174 |            7 |          0.95 |          0.9  |             0.88 |             nan    |           nan    |
| ('openai/gpt-oss-20b', 'datadesc')      | 174 |            2 |          0.98 |          0.97 |             0.96 |             nan    |           nan    |
| ('openai/gpt-oss-20b', 'format')        | 174 |           22 |          0.8  |          0.81 |             0.85 |               0.73 |             0.88 |
| ('openai/gpt-oss-20b', 'format_query')  | 174 |           14 |          0.76 |          0.7  |             0.94 |               0.67 |             0.87 |
| ('qwen/qwen3.6-27b', 'baseline')        | 174 |           11 |          0.8  |          0.77 |             0.94 |             nan    |           nan    |
| ('qwen/qwen3.6-27b', 'datadesc')        | 174 |           10 |          0.83 |          0.96 |             0.85 |             nan    |           nan    |
| ('qwen/qwen3.6-27b', 'format')          | 174 |           28 |          0.79 |          0.82 |             0.9  |               0.73 |             0.79 |
| ('qwen/qwen3.6-27b', 'format_query')    | 174 |           16 |          0.89 |          0.97 |             0.92 |               0.87 |             0.91 |
| ('qwen/qwen3.8-27b', 'baseline')        | 174 |            0 |          0.79 |          0.68 |             0.9  |             nan    |           nan    |
| ('qwen/qwen3.8-27b', 'datadesc')        | 174 |            1 |          0.8  |          0.9  |             0.94 |             nan    |           nan    |
| ('qwen/qwen3.8-27b', 'format')          | 174 |            6 |          0.89 |          0.92 |             0.92 |               0.89 |             0.7  |
| ('qwen/qwen3.8-27b', 'format_query')    | 174 |            8 |          0.84 |          0.91 |             0.9  |               0.89 |             0.92 |

### Where did the stats come from? (filter replies under format configs)

|                                         |   honest |   silent |   unverified |   wrong |
|:----------------------------------------|---------:|---------:|-------------:|--------:|
| ('openai/gpt-oss-120b', 'format')       |       10 |       90 |            0 |       2 |
| ('openai/gpt-oss-120b', 'format_query') |       11 |       78 |            4 |       9 |
| ('openai/gpt-oss-20b', 'format')        |       65 |       28 |            4 |       5 |
| ('openai/gpt-oss-20b', 'format_query')  |       59 |       34 |            0 |       9 |
| ('qwen/qwen3.6-27b', 'format')          |       59 |       27 |            2 |      14 |
| ('qwen/qwen3.6-27b', 'format_query')    |       81 |       13 |            0 |       8 |
| ('qwen/qwen3.8-27b', 'format')          |       64 |       10 |            6 |      22 |
| ('qwen/qwen3.8-27b', 'format_query')    |       85 |       10 |            0 |       7 |

### Vocabulary traps (children / steerage / alone): filter_rows accuracy

| model               |   baseline |   datadesc |   format |   format_query |
|:--------------------|-----------:|-----------:|---------:|---------------:|
| openai/gpt-oss-120b |       0.85 |       1    |     0.37 |           0.56 |
| openai/gpt-oss-20b  |       0.85 |       0.96 |     0.81 |           0.78 |
| qwen/qwen3.6-27b    |       0.78 |       0.89 |     0.78 |           0.96 |
| qwen/qwen3.8-27b    |       0.63 |       0.81 |     0.96 |           0.93 |
