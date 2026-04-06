# Deterministic Checks

33 checks that run locally with zero API calls. Free, instant, no API key needed.

## String Checks

| Check | Description | Example |
|-------|-------------|---------|
| `contains` | Substring present | `check.contains(output, "Python")` |
| `not_contains` | Substring absent | `check.not_contains(output, "error")` |
| `exact_match` | Exact string match | `check.exact_match(output, "expected")` |
| `starts_with` | Prefix match | `check.starts_with(output, "Hello")` |
| `ends_with` | Suffix match | `check.ends_with(output, ".")` |
| `regex` | Regex pattern | `check.regex(output, r"\d{3}-\d{4}")` |
| `similarity` | Levenshtein similarity | `check.similarity(output, expected, threshold=0.8)` |

## Length Checks

| Check | Description | Example |
|-------|-------------|---------|
| `max_tokens` | Max token count | `check.max_tokens(output, limit=200)` |
| `min_tokens` | Min token count | `check.min_tokens(output, minimum=10)` |
| `word_count` | Word count range | `check.word_count(output, min_words=5, max_words=100)` |
| `char_count` | Character count | `check.char_count(output, min_chars=20)` |
| `sentence_count` | Sentence count | `check.sentence_count(output, min_sentences=2)` |

## Structure Checks

| Check | Description | Example |
|-------|-------------|---------|
| `is_json` | Valid JSON | `check.is_json(output)` |
| `is_valid_python` | Valid Python | `check.is_valid_python(output)` |
| `json_schema` | Pydantic schema | `check.json_schema(output, schema=MyModel)` |
| `json_field` | JSON field value | `check.json_field(output, field="name")` |
| `is_valid_sql` | Valid SQL | `check.is_valid_sql(output)` |
| `is_valid_markdown` | Valid Markdown | `check.is_valid_markdown(output)` |

## Quality Checks

| Check | Description | Example |
|-------|-------------|---------|
| `readability` | Flesch-Kincaid grade | `check.readability(output, max_grade=8.0)` |
| `language` | Language detection | `check.language(output, expected="en")` |
| `bleu` | BLEU score | `check.bleu(output, reference=expected)` |
| `rouge_l` | ROUGE-L score | `check.rouge_l(output, reference=expected)` |

## Compound Checks

| Check | Description | Example |
|-------|-------------|---------|
| `all_of` | All substrings | `check.all_of(output, ["A", "B", "C"])` |
| `any_of` | Any substring | `check.any_of(output, ["A", "B", "C"])` |
| `none_of` | No substrings | `check.none_of(output, ["error", "null"])` |

## Safety

| Check | Description | Example |
|-------|-------------|---------|
| `no_pii` | PII detection | `check.no_pii(output)` |

## Numeric

| Check | Description | Example |
|-------|-------------|---------|
| `greater_than` | Numeric > value | `check.greater_than("Score: 85", threshold=70)` |
| `less_than` | Numeric < value | `check.less_than("Latency: 120ms", threshold=200)` |
| `between` | Numeric in range | `check.between("Confidence: 0.87", low=0.0, high=1.0)` |

## Performance

| Check | Description | Example |
|-------|-------------|---------|
| `latency` | Response time | `check.latency(response_ms, max_ms=2000)` |
| `cost` | API cost | `check.cost(cost_usd, max_usd=0.05)` |
