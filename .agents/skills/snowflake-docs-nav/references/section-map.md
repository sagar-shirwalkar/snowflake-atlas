# Snowflake Docs Section Map

Complete mapping of all sections from the root `llms.txt` with page counts and URLs.

## Root Index
- **URL**: `https://docs.snowflake.com/llms.txt`
- **Sections**: 30 top-level entries

## Section Details

| Section | llms.txt URL | Pages | Description |
|---------|--------------|-------|-------------|
| General / Reference | `reference.md` | ~1 | Top-level landing pages |
| User Guide | `user-guide/llms.txt` | 881 | Core product docs |
| Loading & Unloading Data | `user-guide/data-integration/llms.txt` | 682 | Stages, COPY, Snowpipe, Openflow |
| Snowflake Cortex (AI/ML) | `user-guide/snowflake-cortex/llms.txt` | 102 | LLM functions, vector search, Document AI |
| Cortex Code | `user-guide/cortex-code/llms.txt` | 54 | AI coding agent |
| Clean Rooms | `user-guide/cleanrooms/llms.txt` | 113 | Data clean rooms |
| Snowsight UI | `user-guide/ui-snowsight/llms.txt` | 46 | Web interface |
| Snowflake Postgres | `user-guide/snowflake-postgres/llms.txt` | 25 | Managed Postgres |
| SQL Functions | `sql-reference/functions/llms.txt` | 994 | All built-in functions |
| SQL Commands | `sql-reference/sql/llms.txt` | 676 | DDL/DML commands |
| Account Usage | `sql-reference/account-usage/llms.txt` | 174 | ACCOUNT_USAGE views |
| Organization Usage | `sql-reference/organization-usage/llms.txt` | 95 | ORGANIZATION_USAGE views |
| Information Schema | `sql-reference/info-schema/llms.txt` | 60 | INFO_SCHEMA views |
| SQL Classes | `sql-reference/classes/llms.txt` | 118 | ML classes (FORECAST, ANOMALY_DETECTION, etc.) |
| SQL General Reference | `sql-reference/llms.txt` | 239 | Parameters, data types, scripting |
| Connectors & Drivers | `connectors/llms.txt` | 107 | Kafka, Google, Microsoft, etc. |
| Collaboration & Marketplace | `collaboration/llms.txt` | 73 | Marketplace, data sharing |
| Migrations | `migrations/llms.txt` | 742 | Migration guides |
| Release Notes | `release-notes/llms.txt` | 1666 | Version history |
| Programmatic Access | `progaccess/llms.txt` | 3 | REST APIs |
| Developer Guide | `developer-guide/llms.txt` | 317 | UDFs, SPs, drivers, extensibility |
| Snowpark | `developer-guide/snowpark/llms.txt` | 67 | Python/Java/Scala API |
| Snowflake ML | `developer-guide/snowflake-ml/llms.txt` | 84 | ML APIs, feature store, model registry |
| Native Apps Framework | `developer-guide/native-apps/llms.txt` | 143 | Building data apps |
| Streamlit in Snowflake | `developer-guide/streamlit/llms.txt` | 34 | Streamlit apps |
| Snowflake CLI | `developer-guide/snowflake-cli/llms.txt` | 244 | CLI for CI/CD |
| Snowpark Container Services | `developer-guide/snowpark-container-services/llms.txt` | 34 | Container workloads |
| Snowflake REST API | `developer-guide/snowflake-rest-api/llms.txt` | 52 | REST API reference |

## Total Pages: ~6,800+

## URL Patterns

```
Root:           https://docs.snowflake.com/llms.txt
Section:        https://docs.snowflake.com/en/{section-path}/llms.txt
Page:           https://docs.snowflake.com/en/{section-path}/{page-name}.md
```

## Notes

- All pages are `.md` (markdown) files
- No `.txt` files referencing `.md` — the user's concern doesn't apply to current Snowflake docs
- Frontmatter varies: some pages have full YAML (title, description, product_area, last_updated), others have minimal or none
- Section `none
- Rate limit: undocumented; be respectful (add delays between requests)