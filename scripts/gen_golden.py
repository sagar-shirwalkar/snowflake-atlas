#!/usr/bin/env python3
"""Expand the golden evaluation set to 200 queries.

Scans `data/snowflake-docs/markdown/`, groups files by topic,
generates natural-language queries with expected files, and
appends to `data/golden_set.jsonl`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
MARKDOWN_ROOT = Path("data/snowflake-docs/markdown")
GOLDEN_PATH = Path("data/golden_set.jsonl")
TARGET = 200
SEED = 42

random.seed(SEED)


def load_existing(path: Path) -> set[str]:
    """Return set of queries already in the golden set."""
    if not path.is_file():
        return set()
    queries = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    queries.add(json.loads(line)["query"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return queries


def get_markdown_files(root: Path) -> dict[str, list[Path]]:
    """Return all .md files grouped by their directory."""
    groups: dict[str, list[Path]] = {"root": []}
    for p in sorted(root.rglob("*.md")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) > 1 and parts[-2] != "markdown":
            group = str(Path(*parts[:-1]))
        else:
            group = "root"
        groups.setdefault(group, []).append(p)
    return groups


# ── Query generators ──────────────────────────────────────────

TOPIC_QUERIES: list[tuple[str, list[str], str]] = [
    # (query, [expected_file_paths_relative_to_markdown_root], topic_description)

    # ── Warehouses ──
    ("how do i create and configure a virtual warehouse",
     ["user-guide/warehouses.md", "user-guide/warehouses-tasks.md",
      "user-guide/warehouses-considerations.md"], "warehouses"),
    ("what are the different warehouse sizes and how do they affect performance",
     ["user-guide/warehouses-overview.md", "user-guide/performance-query-warehouse-size.md",
      "user-guide/warehouses.md"], "warehouses"),
    ("multi-cluster warehouse setup for concurrency",
     ["user-guide/warehouses-multicluster.md", "user-guide/warehouses.md",
      "user-guide/warehouses-tasks.md"], "warehouses"),
    ("how to set up adaptive warehouse scaling",
     ["user-guide/warehouses-adaptive.md", "user-guide/warehouses.md"], "warehouses"),
    ("monitoring warehouse load and query performance",
     ["user-guide/warehouses-load-monitoring.md", "user-guide/performance-query-warehouse.md"], "warehouses"),

    # ── Data Loading ──
    ("bulk loading from s3 using copy into",
     ["user-guide/data-load-s3.md", "user-guide/data-load-s3-config.md",
      "user-guide/data-load-s3-copy.md", "user-guide/data-load-s3-create-stage.md"], "data-load"),
    ("loading data from azure blob storage",
     ["user-guide/data-load-azure.md", "user-guide/data-load-azure-config.md",
      "user-guide/data-load-azure-copy.md", "user-guide/data-load-azure-create-stage.md"], "data-load"),
    ("loading data from google cloud storage",
     ["user-guide/data-load-gcs.md", "user-guide/data-load-gcs-config.md",
      "user-guide/data-load-gcs-copy.md"], "data-load"),
    ("loading data from local filesystem",
     ["user-guide/data-load-local-file-system.md", "user-guide/data-load-local-file-system-stage.md",
      "user-guide/data-load-local-file-system-copy.md"], "data-load"),
    ("data loading best practices and considerations",
     ["user-guide/data-load-considerations.md", "user-guide/data-load-considerations-plan.md",
      "user-guide/data-load-considerations-prepare.md", "user-guide/data-load-considerations-stage.md"], "data-load"),
    ("transforming data during loading",
     ["user-guide/data-load-transform.md", "user-guide/data-load-prepare.md"], "data-load"),
    ("monitoring data loads in snowflake",
     ["user-guide/data-load-monitor.md", "user-guide/data-load-considerations-manage.md"], "data-load"),

    # ── Snowpipe ──
    ("automating continuous data ingestion with snowpipe",
     ["user-guide/data-load-snowpipe-intro.md", "user-guide/data-load-snowpipe-auto.md",
      "user-guide/data-load-snowpipe-auto-s3.md"], "snowpipe"),
    ("setting up snowpipe auto-ingest for azure",
     ["user-guide/data-load-snowpipe-auto-azure.md", "user-guide/data-load-snowpipe-intro.md"], "snowpipe"),
    ("snowpipe rest api for programmatic loading",
     ["user-guide/data-load-snowpipe-rest-overview.md", "user-guide/data-load-snowpipe-rest-apis.md",
      "user-guide/data-load-snowpipe-rest-load.md", "user-guide/data-load-snowpipe-rest-gs.md"], "snowpipe"),
    ("billing and cost considerations for snowpipe",
     ["user-guide/data-load-snowpipe-billing.md", "user-guide/data-load-snowpipe-intro.md"], "snowpipe"),
    ("snowpipe error handling and troubleshooting",
     ["user-guide/data-load-snowpipe-errors.md", "user-guide/data-load-snowpipe-errors-sns.md",
      "user-guide/data-load-snowpipe-errors-azure.md"], "snowpipe"),

    # ── Snowpipe Streaming ──
    ("snowpipe streaming overview and architecture",
     ["user-guide/snowpipe-streaming/data-load-snowpipe-streaming-overview.md",
      "user-guide/snowpipe-streaming/snowpipe-streaming-classic-overview.md"], "snowpipe-streaming"),
    ("high performance snowpipe streaming ingestion",
     ["user-guide/snowpipe-streaming/snowpipe-streaming-high-performance-overview.md",
      "user-guide/snowpipe-streaming/snowpipe-streaming-high-performance-getting-started.md",
      "user-guide/snowpipe-streaming/snowpipe-streaming-high-performance-configurations.md"], "snowpipe-streaming"),
    ("snowpipe streaming error tables and handling",
     ["user-guide/snowpipe-streaming/snowpipe-streaming-error-tables.md",
      "user-guide/snowpipe-streaming/snowpipe-streaming-high-performance-error-handling.md"], "snowpipe-streaming"),
    ("migrating from classic to high performance snowpipe streaming",
     ["user-guide/snowpipe-streaming/snowpipe-streaming-high-performance-migration.md",
      "user-guide/snowpipe-streaming/snowpipe-streaming-classic-overview.md"], "snowpipe-streaming"),
    ("snowpipe streaming iceberg table support",
     ["user-guide/snowpipe-streaming/snowpipe-streaming-classic-iceberg.md",
      "user-guide/snowpipe-streaming/snowpipe-streaming-high-performance-iceberg.md"], "snowpipe-streaming"),

    # ── Dynamic Tables ──
    ("dynamic tables refresh modes incremental vs full",
     ["user-guide/dynamic-tables/refresh-modes.md", "user-guide/dynamic-tables/overview.md",
      "user-guide/dynamic-tables/refresh-optimization.md"], "dynamic-tables"),
    ("managing and monitoring dynamic tables",
     ["user-guide/dynamic-tables/manage.md", "user-guide/dynamic-tables/monitoring.md",
      "user-guide/dynamic-tables/target-lag.md"], "dynamic-tables"),
    ("dynamic tables best practices and design patterns",
     ["user-guide/dynamic-tables/best-practices.md", "user-guide/dynamic-tables/design-patterns.md",
      "user-guide/dynamic-tables/decision-guide.md"], "dynamic-tables"),
    ("dynamic tables cost and warehouse selection",
     ["user-guide/dynamic-tables/cost.md", "user-guide/dynamic-tables/warehouse-selection.md",
      "user-guide/dynamic-tables/input-data-optimization.md"], "dynamic-tables"),
    ("migrating from streams and tasks to dynamic tables",
     ["user-guide/dynamic-tables/migrate-streams-tasks.md", "user-guide/dynamic-tables/overview.md",
      "user-guide/dynamic-tables/streams-on-dts.md"], "dynamic-tables"),
    ("dynamic tables data consistency and cloning",
     ["user-guide/dynamic-tables/data-consistency.md", "user-guide/dynamic-tables/cloning.md",
      "user-guide/dynamic-tables/frozen-regions.md"], "dynamic-tables"),
    ("sharing and replication of dynamic tables",
     ["user-guide/dynamic-tables/sharing.md", "user-guide/dynamic-tables/replication.md",
      "user-guide/dynamic-tables/storage-lifecycle-policies.md"], "dynamic-tables"),
    ("creating dynamic tables on iceberg tables",
     ["user-guide/dynamic-tables/create-iceberg.md", "user-guide/dynamic-tables/create.md"], "dynamic-tables"),
    ("troubleshooting dynamic table creation and refreshes",
     ["user-guide/dynamic-tables/troubleshoot-creation.md", "user-guide/dynamic-tables/troubleshoot-permissions.md",
      "user-guide/dynamic-tables/troubleshoot-refreshes.md"], "dynamic-tables"),
    ("dynamic tables supported query types",
     ["user-guide/dynamic-tables/supported-queries.md", "user-guide/dynamic-tables/custom-incrementalization.md",
      "user-guide/dynamic-tables/dbt.md"], "dynamic-tables"),

    # ── Streams and Tasks ──
    ("introduction to streams in snowflake",
     ["user-guide/streams-intro.md", "user-guide/streams-manage.md", "user-guide/streams-examples.md"], "streams-tasks"),
    ("creating and managing task graphs",
     ["user-guide/tasks-intro.md", "user-guide/tasks-graphs.md", "user-guide/tasks-monitor.md"], "streams-tasks"),
    ("scheduling tasks with cron triggers",
     ["user-guide/tasks-triggered.md", "user-guide/tasks-intro.md"], "streams-tasks"),
    ("python and java based tasks in snowflake",
     ["user-guide/tasks-python-jvm.md", "user-guide/stored-procedure/python.md"], "streams-tasks"),
    ("task error handling and notifications",
     ["user-guide/tasks-errors.md", "user-guide/tasks-errors-integrate.md",
      "user-guide/tasks-success-integrate.md", "user-guide/tasks-events.md"], "streams-tasks"),

    # ── Access Control & Security ──
    ("snowflake role based access control overview",
     ["user-guide/security-access-control-overview.md", "user-guide/security-access-control-configure.md",
      "user-guide/security-access-control-considerations.md"], "security"),
    ("privileges available in snowflake",
     ["user-guide/security-access-control-privileges.md", "user-guide/security-access-control-overview.md"], "security"),
    ("authentication methods in snowflake",
     ["user-guide/security-authentication-overview.md", "user-guide/password-authentication.md",
      "user-guide/key-pair-auth.md"], "security"),
    ("multi-factor authentication setup",
     ["user-guide/security-mfa.md", "user-guide/security-mfa-second-factor.md",
      "user-guide/security-mfa-duo.md", "user-guide/security-mfa-rollout.md"], "security"),
    ("scim provisioning for identity federation",
     ["user-guide/scim-intro.md", "user-guide/scim-okta.md", "user-guide/scim-azure.md",
      "user-guide/scim-api-references.md"], "security"),
    ("network policies and ip allowlisting",
     ["user-guide/network-policies.md", "user-guide/network-policy-advisor.md",
      "user-guide/network-rules.md", "user-guide/hostname-allowlist.md"], "security"),
    ("oauth integration with snowflake",
     ["user-guide/oauth-intro.md", "user-guide/oauth-snowflake-overview.md",
      "user-guide/oauth-okta.md", "user-guide/oauth-azure.md"], "security"),
    ("row access policies for data governance",
     ["user-guide/security-row-intro.md", "user-guide/security-row-using.md",
      "user-guide/security-access-control-privileges.md"], "security"),
    ("column level security with dynamic data masking",
     ["user-guide/security-column-intro.md", "user-guide/security-column-ddm-intro.md",
      "user-guide/security-column-ddm-use.md"], "security"),
    ("encryption key management in snowflake",
     ["user-guide/security-encryption-manage.md", "user-guide/security-encryption-tss.md",
      "user-guide/security-encryption-end-to-end.md"], "security"),

    # ── Data Sharing ──
    ("introduction to data sharing in snowflake",
     ["user-guide/data-sharing-intro.md", "user-guide/data-sharing-gs.md"], "data-sharing"),
    ("setting up as a data provider",
     ["user-guide/data-sharing-provider.md", "user-guide/data-sharing-secure-views.md",
      "user-guide/data-sharing-views.md"], "data-sharing"),
    ("consuming shared data as a reader account",
     ["user-guide/data-sharing-reader-config.md", "user-guide/data-sharing-reader-create.md",
      "user-guide/data-share-consumers.md"], "data-sharing"),
    ("data marketplace and exchange",
     ["user-guide/data-marketplace.md", "user-guide/data-exchange.md",
      "user-guide/data-exchange-becoming-a-provider.md",
      "user-guide/data-exchange-managing-data-listings.md"], "data-sharing"),
    ("sharing data across regions and platforms",
     ["user-guide/secure-data-sharing-across-regions-platforms.md",
      "user-guide/data-sharing-intro.md"], "data-sharing"),

    # ── Tables ──
    ("hybrid tables for operational workloads",
     ["user-guide/tables-hybrid.md", "user-guide/tables-hybrid-create.md",
      "user-guide/tables-hybrid-best-practices.md",
      "user-guide/hybrid-tables-operational-query-performance.md"], "tables"),
    ("external tables on s3 azure gcs",
     ["user-guide/tables-external-intro.md", "user-guide/tables-external-s3.md",
      "user-guide/tables-external-azure.md", "user-guide/tables-external-gcs.md"], "tables"),
    ("temporary and transient tables",
     ["user-guide/tables-temp-transient.md", "user-guide/table-considerations.md"], "tables"),
    ("understanding micro partitions and data clustering",
     ["user-guide/tables-clustering-micropartitions.md", "user-guide/tables-clustering-keys.md",
      "user-guide/tables-micro-partitions.md"], "tables"),
    ("auto clustering and manual clustering",
     ["user-guide/tables-auto-reclustering.md", "user-guide/tables-clustering-manual.md",
      "user-guide/tables-clustering-keys.md"], "tables"),

    # ── Iceberg Tables ──
    ("iceberg tables overview in snowflake",
     ["user-guide/tables-iceberg.md", "user-guide/tables-iceberg-create.md",
      "user-guide/tables-iceberg-manage.md"], "iceberg"),
    ("configuring iceberg catalog integrations",
     ["user-guide/tables-iceberg-configure-catalog-integration.md",
      "user-guide/tables-iceberg-configure-catalog-integration-glue.md",
      "user-guide/tables-iceberg-configure-catalog-integration-rest.md"], "iceberg"),
    ("iceberg external volumes for cloud storage",
     ["user-guide/tables-iceberg-configure-external-volume.md",
      "user-guide/tables-iceberg-configure-external-volume-s3.md",
      "user-guide/tables-iceberg-configure-external-volume-azure.md",
      "user-guide/tables-iceberg-managing-external-volumes.md"], "iceberg"),
    ("querying iceberg tables with external engines",
     ["user-guide/tables-iceberg-query-using-external-query-engine-snowflake-horizon.md",
      "user-guide/tables-iceberg-access-using-external-query-engine-snowflake-horizon.md",
      "user-guide/tables-iceberg-open-catalog-query.md",
      "user-guide/tables-iceberg-use-external-query-engine.md"], "iceberg"),
    ("iceberg table data types and storage considerations",
     ["user-guide/tables-iceberg-data-types.md", "user-guide/tables-iceberg-storage.md",
      "user-guide/tables-iceberg-internal-storage.md",
      "user-guide/tables-iceberg-default-metadata-format.md"], "iceberg"),

    # ── Data Quality ──
    ("snowflake data quality monitoring overview",
     ["user-guide/data-quality-intro.md", "user-guide/data-quality-monitor.md",
      "user-guide/data-quality-expectations.md"], "data-quality"),
    ("setting up data quality monitors in the ui",
     ["user-guide/data-quality-ui-setup.md", "user-guide/data-quality-ui-monitor.md",
      "user-guide/data-quality-results.md"], "data-quality"),
    ("data quality anomaly detection",
     ["user-guide/data-quality-anomaly.md", "user-guide/data-quality-filter.md",
      "user-guide/data-quality-fixing.md"], "data-quality"),
    ("data quality schema level monitors and notifications",
     ["user-guide/data-quality-schema-level.md", "user-guide/data-quality-notifications.md",
      "user-guide/data-quality-group-by.md"], "data-quality"),

    # ── Cost Management ──
    ("understanding snowflake compute costs",
     ["user-guide/cost-understanding-compute.md", "user-guide/cost-optimize.md",
      "user-guide/cost-exploring-compute.md"], "cost"),
    ("storage costs and data lifecycle",
     ["user-guide/cost-understanding-data-storage.md", "user-guide/cost-exploring-data-storage.md",
      "user-guide/data-cdp-storage-costs.md"], "cost"),
    ("data transfer costs across clouds and regions",
     ["user-guide/cost-understanding-data-transfer.md", "user-guide/cost-exploring-data-transfer.md",
      "user-guide/cost-optimize-cloud-services.md"], "cost"),
    ("cost management dashboards and insights",
     ["user-guide/cost-insights.md", "user-guide/cost-management-overview.md",
      "user-guide/cost-exploring-overall.md"], "cost"),
    ("controlling costs with resource monitors and policies",
     ["user-guide/cost-controlling.md", "user-guide/cost-controlling-controls.md",
      "user-guide/resource-monitors.md"], "cost"),
    ("anomaly detection for cost spikes",
     ["user-guide/cost-anomalies.md", "user-guide/cost-anomalies-ui.md",
      "user-guide/cost-anomalies-class.md"], "cost"),

    # ── Budgets ──
    ("setting account level budgets in snowflake",
     ["user-guide/budgets/account-budget.md", "user-guide/budgets/custom-budget.md",
      "user-guide/budgets/cost.md"], "budgets"),
    ("budget notifications and custom actions",
     ["user-guide/budgets/notifications.md", "user-guide/budgets/custom-actions.md",
      "user-guide/budgets/cycle-start-actions.md"], "budgets"),
    ("monitoring budget usage and shared resources",
     ["user-guide/budgets/monitor.md", "user-guide/budgets/budget-shared-resources.md",
      "user-guide/budgets/troubleshoot.md"], "budgets"),

    # ── Time Travel ──
    ("using time travel to query historical data",
     ["user-guide/data-time-travel.md", "user-guide/data-availability.md"], "time-travel"),
    ("failsafe and data recovery",
     ["user-guide/data-failsafe.md", "user-guide/data-availability.md",
      "user-guide/data-lifecycle.md"], "time-travel"),

    # ── Views ──
    ("views materialized views and dynamic tables comparison",
     ["user-guide/overview-view-mview-dts.md", "user-guide/views-introduction.md",
      "user-guide/views-materialized.md"], "views"),
    ("secure views for data protection",
     ["user-guide/views-secure.md", "user-guide/data-sharing-secure-views.md"], "views"),

    # ── Querying ──
    ("optimizing query performance with options",
     ["user-guide/performance-query-options.md", "user-guide/performance-query-warehouse.md",
      "user-guide/performance-query-storage.md"], "querying"),
    ("using the query acceleration service",
     ["user-guide/query-acceleration-service.md", "user-guide/performance-query-warehouse-queue.md"], "querying"),
    ("ctes and subqueries in snowflake",
     ["user-guide/queries-cte.md", "user-guide/querying-subqueries.md"], "querying"),
    ("querying semi structured json data",
     ["user-guide/querying-semistructured.md", "user-guide/semistructured-intro.md",
      "user-guide/semistructured-considerations.md"], "querying"),
    ("approximate query functions for cardinality and percentiles",
     ["user-guide/querying-approximate-cardinality.md",
      "user-guide/querying-approximate-percentile-values.md",
      "user-guide/querying-approximate-frequent-values.md",
      "user-guide/querying-approximate-similarity.md"], "querying"),
    ("using match recognize for pattern matching",
     ["user-guide/match-recognize-introduction.md"], "querying"),
    ("time series data analysis in snowflake",
     ["user-guide/querying-time-series-data.md"], "querying"),
    ("hierarchical queries with connect by",
     ["user-guide/queries-hierarchical.md"], "querying"),

    # ── Kafka Connector ──
    ("kafka connector overview and how it works",
     ["user-guide/kafka-connector/index.md", "user-guide/kafka-connector/about-kafka-connect.md",
      "user-guide/kafka-connector/how-the-connector-works.md"], "kafka"),
    ("setting up the kafka connector on snowflake side",
     ["user-guide/kafka-connector/setup-snowflake.md", "user-guide/kafka-connector/setup-kafka.md",
      "user-guide/kafka-connector/setup-tasks.md"], "kafka"),
    ("kafka connector monitoring and troubleshooting",
     ["user-guide/kafka-connector/monitor.md", "user-guide/kafka-connector/troubleshoot.md",
      "user-guide/kafka-connector/validation-error-handling.md"], "kafka"),
    ("migrating kafka connector from v3 to v4",
     ["user-guide/kafka-connector/migrate-v3-to-v4.md",
      "user-guide/kafka-connector/index.md"], "kafka"),

    # ── Account Admin / Organizations ──
    ("account identifier formats and usage",
     ["user-guide/admin-account-identifier.md", "user-guide/admin-account-management.md"], "admin"),
    ("organization management and multi account setup",
     ["user-guide/organizations.md", "user-guide/organization-accounts.md",
      "user-guide/organizations-manage-accounts.md"], "admin"),
    ("account replication and failover across regions",
     ["user-guide/account-replication-intro.md", "user-guide/account-replication-config.md",
      "user-guide/account-replication-considerations.md",
      "user-guide/account-replication-failover-failback.md"], "admin"),
    ("database replication across accounts",
     ["user-guide/db-replication-intro.md", "user-guide/db-replication-config.md",
      "user-guide/database-replication-considerations.md"], "admin"),

    # ── Classification and Governance ──
    ("automatic data classification in snowflake",
     ["user-guide/classify-auto.md", "user-guide/classify-intro.md",
      "user-guide/classify-native.md", "user-guide/classify-results.md"], "governance"),
    ("custom classification with tagging",
     ["user-guide/classify-custom.md", "user-guide/classify-custom-using.md",
      "user-guide/object-tagging.md"], "governance"),
    ("aggregation policies for entity privacy",
     ["user-guide/aggregation-policies.md", "user-guide/aggregation-policies-entity-privacy.md"], "governance"),
    ("projection policies for column visibility",
     ["user-guide/projection-policies.md", "user-guide/security-column-intro.md"], "governance"),
    ("tag based masking policies",
     ["user-guide/tag-based-masking-policies.md", "user-guide/security-column-ddm-intro.md"], "governance"),

    # ── Snowflake Copilot ──
    ("using snowflake copilot for sql generation",
     ["user-guide/snowflake-copilot.md", "user-guide/snowflake-copilot-inline.md"], "copilot"),

    # ── Snowsight UI ──
    ("getting started with snowsight",
     ["user-guide/ui-snowsight.md", "user-guide/ui-snowsight-gs.md",
      "user-guide/ui-snowsight-navigation.md"], "snowsight"),
    ("creating dashboards and visualizations in snowsight",
     ["user-guide/ui-snowsight-dashboards.md", "user-guide/ui-snowsight-visualizations.md",
      "user-guide/ui-snowsight-filters.md"], "snowsight"),
    ("worksheets in snowsight",
     ["user-guide/ui-snowsight-worksheets.md", "user-guide/ui-snowsight-worksheets-gs.md",
      "user-guide/ui-snowsight-query.md"], "snowsight"),

    # ── SnowSQL and SFSQL ──
    ("installing and configuring snowsql",
     ["user-guide/snowsql-install-config.md", "user-guide/snowsql-config.md",
      "user-guide/snowsql-start.md", "user-guide/snowsql-use.md"], "snowsql"),
    ("snowflake sfsql client usage",
     ["user-guide/sfsql.md", "user-guide/sfsql-install-config.md",
      "user-guide/sfsql-start-stop.md", "user-guide/sfsql-use.md"], "snowsql"),
    ("migrating from snowsql to sfsql",
     ["user-guide/snowsql-migrate.md", "user-guide/snowsql-sfsql-diff.md"], "snowsql"),

    # ── Private Connectivity ──
    ("privatelink setup for snowflake on azure",
     ["user-guide/privatelink-azure.md", "user-guide/admin-security-privatelink.md",
      "user-guide/private-connectivity-inbound.md"], "networking"),
    ("private connectivity for aws snowflake",
     ["user-guide/private-internal-stages-aws.md", "user-guide/private-manage-endpoints-aws.md",
      "user-guide/private-managed-volumes-aws.md"], "networking"),
    ("private connectivity for gcp snowflake",
     ["user-guide/private-internal-stages-gcp.md", "user-guide/private-manage-endpoints-gcp.md",
      "user-guide/private-service-connect-google.md"], "networking"),

    # ── Developer Guide: Snowflake ML ──
    ("snowflake ml overview and quickstart",
     ["developer-guide/snowflake-ml/overview.md", "developer-guide/snowflake-ml/quickstart.md",
      "developer-guide/snowflake-ml/snowpark-ml.md"], "ml"),
    ("model registry in snowflake ml",
     ["developer-guide/snowflake-ml/model-registry.md",
      "developer-guide/snowflake-ml/modeling.md",
      "developer-guide/snowflake-ml/train-models.md"], "ml"),
    ("feature store for ml pipelines",
     ["developer-guide/snowflake-ml/feature-store.md",
      "developer-guide/snowflake-ml/load-data.md",
      "developer-guide/snowflake-ml/transform-data.md"], "ml"),
    ("distributed training in snowflake ml",
     ["developer-guide/snowflake-ml/distributed-training.md",
      "developer-guide/snowflake-ml/train-models-across-partitions.md",
      "developer-guide/snowflake-ml/process-data-across-partitions.md"], "ml"),
    ("notebooks in snowflake container services",
     ["developer-guide/snowflake-ml/notebooks-on-spcs.md",
      "developer-guide/snowflake-ml/experiments.md",
      "developer-guide/snowflake-ml/agentic-ml.md"], "ml"),
    ("ml jobs and container runtime",
     ["developer-guide/snowflake-ml/ml-jobs.md",
      "developer-guide/snowflake-ml/container-runtime-ml.md",
      "developer-guide/snowflake-ml/container-runtime-multi-node.md"], "ml"),
    ("ml lineage and model monitoring",
     ["developer-guide/snowflake-ml/ml-lineage.md",
      "developer-guide/snowflake-ml/create-pipelines-deploy.md",
      "developer-guide/snowflake-ml/dataset.md"], "ml"),

    # ── Developer Guide: Python API ──
    ("snowflake python api overview and installation",
     ["developer-guide/snowflake-python-api/snowflake-python-overview.md",
      "developer-guide/snowflake-python-api/snowflake-python-installing.md",
      "developer-guide/snowflake-python-api/snowflake-python-general-concepts.md"], "python-api"),
    ("connecting to snowflake with python api",
     ["developer-guide/snowflake-python-api/snowflake-python-connecting-snowflake.md",
      "developer-guide/snowflake-python-api/overview-tutorials.md"], "python-api"),
    ("managing warehouses with python api",
     ["developer-guide/snowflake-python-api/snowflake-python-managing-warehouses.md",
      "developer-guide/snowflake-python-api/snowflake-python-managing-databases.md"], "python-api"),
    ("managing users and roles with python api",
     ["developer-guide/snowflake-python-api/snowflake-python-managing-user-roles.md",
      "developer-guide/snowflake-python-api/snowflake-python-managing-network-policies.md"], "python-api"),
    ("managing dynamic tables and streams with python api",
     ["developer-guide/snowflake-python-api/snowflake-python-managing-dynamic-tables.md",
      "developer-guide/snowflake-python-api/snowflake-python-managing-streams.md",
      "developer-guide/snowflake-python-api/snowflake-python-managing-tasks.md"], "python-api"),
    ("managing alerts and tags with python api",
     ["developer-guide/snowflake-python-api/snowflake-python-managing-alerts.md",
      "developer-guide/snowflake-python-api/snowflake-python-managing-tags.md"], "python-api"),

    # ── Developer Guide: Functions & Procedures ──
    ("stored procedures vs udfs comparison",
     ["developer-guide/stored-procedures-vs-udfs.md",
      "developer-guide/udf-stored-procedure-guidelines.md",
      "developer-guide/udf-stored-procedure-arguments.md"], "sp-udf"),
    ("building udfs and stored procedures with maven and sbt",
     ["developer-guide/udf-stored-procedure-building.md",
      "developer-guide/udf-stored-procedure-build-maven.md",
      "developer-guide/udf-stored-procedure-build-sbt.md"], "sp-udf"),
    ("naming conventions and security for udfs",
     ["developer-guide/udf-stored-procedure-naming-conventions.md",
      "developer-guide/udf-stored-procedure-security-practices.md",
      "developer-guide/udf-stored-procedure-constraints.md"], "sp-udf"),

    # ── SQL Reference ──
    ("ddl for virtual warehouses",
     ["sql-reference/ddl-virtual-warehouse.md", "sql-reference/sql/create-warehouse.md",
      "sql-reference/sql/alter-warehouse.md"], "sql"),
    ("data types in snowflake numeric text datetime",
     ["sql-reference/data-types-numeric.md", "sql-reference/data-types-text.md",
      "sql-reference/data-types-datetime.md", "sql-reference/data-types-logical.md"], "sql"),
    ("semistructured and structured data types",
     ["sql-reference/data-types-semistructured.md", "sql-reference/data-types-structured.md",
      "sql-reference/data-types-vector.md"], "sql"),
    ("snowflake ddl for databases and schemas",
     ["sql-reference/ddl-database.md", "sql-reference/ddl-table.md",
      "sql-reference/ddl-stage.md"], "sql"),
    ("snowflake ddl for user security and pipelines",
     ["sql-reference/ddl-user-security.md", "sql-reference/ddl-pipeline.md",
      "sql-reference/ddl-other.md"], "sql"),
    ("collation and locale support in snowflake",
     ["sql-reference/collation.md", "sql-reference/collation-locales.md"], "sql"),
    ("constraints in snowflake create alter drop",
     ["sql-reference/constraints-overview.md", "sql-reference/constraints-create.md",
      "sql-reference/constraints-alter.md", "sql-reference/constraints-drop.md"], "sql"),
    ("snowflake transactions and isolation",
     ["sql-reference/transactions.md", "sql-reference/session-variables.md",
      "sql-reference/ternary-logic.md"], "sql"),
    ("snowflake parameters reference",
     ["sql-reference/parameters.md", "sql-reference/metadata.md",
      "sql-reference/identifier-literal.md"], "sql"),
    ("string and numeric functions reference",
     ["sql-reference/functions-string.md", "sql-reference/functions-numeric.md",
      "sql-reference/functions-aggregation.md"], "sql"),
    ("window functions and analytic queries",
     ["sql-reference/functions-window.md", "sql-reference/functions-window-syntax.md",
      "sql-reference/functions-table.md"], "sql"),
    ("date time functions and examples",
     ["sql-reference/functions-date-time.md", "sql-reference/date-time-input-output.md",
      "sql-reference/date-time-examples.md"], "sql"),
    ("semi structured functions for json handling",
     ["sql-reference/functions-semistructured.md", "sql-reference/functions-conversion.md"], "sql"),
    ("search optimization and model monitoring functions",
     ["sql-reference/search-optimization.md", "sql-reference/functions-model-monitors.md"], "sql"),
    ("geospatial data types and functions",
     ["sql-reference/data-types-geospatial.md", "sql-reference/functions-geospatial.md"], "sql"),
    ("user defined data types and uuid support",
     ["sql-reference/data-types-user-defined.md", "sql-reference/data-types-uuid.md"], "sql"),

    # ── Cortex AI ──
    ("cortex search service overview and usage",
     ["user-guide/snowflake-cortex/cortex-search/cortex-search-overview.md",
      "user-guide/snowflake-cortex/cortex-search/query-cortex-search-service.md"], "cortex-ai"),
    ("cortex search customization costs and monitoring",
     ["user-guide/snowflake-cortex/cortex-search/cortex-search-customize-scoring.md",
      "user-guide/snowflake-cortex/cortex-search/cortex-search-costs.md",
      "user-guide/snowflake-cortex/cortex-search/cortex-search-monitor.md"], "cortex-ai"),
    ("cortex ai llm functions complete sentiment translate",
     ["sql-reference/functions/complete-snowflake-cortex.md",
      "sql-reference/functions/sentiment-snowflake-cortex.md",
      "sql-reference/functions/translate-snowflake-cortex.md",
      "sql-reference/functions/summarize-snowflake-cortex.md"], "cortex-ai"),
    ("cortex ai agentic features and run sdk",
     ["user-guide/cortex-code-agent-sdk/cortex-code-agent-sdk.md",
      "user-guide/cortex-code-agent-sdk/quickstart.md",
      "sql-reference/functions/agent_run-snowflake-cortex.md"], "cortex-ai"),
    ("cortex code cli for ai assisted development",
     ["user-guide/cortex-code/cortex-code-cli.md", "user-guide/cortex-code/cli-reference.md",
      "user-guide/cortex-code/bundled-skills.md"], "cortex-ai"),
    ("cortex code desktop features and navigation",
     ["user-guide/cortex-code/cortex-code-desktop/cortex-code-desktop-usage-history-view.md",
      "user-guide/cortex-code/cortex-code-desktop/navigation.md",
      "user-guide/cortex-code/cortex-code-desktop/agents.md"], "cortex-ai"),

    # ── Object Tagging and Policies ──
    ("object tagging for governance",
     ["user-guide/object-tagging.md", "user-guide/tag-based-masking-policies.md"], "tagging"),
    ("session policies and authentication policies",
     ["user-guide/session-policies.md", "user-guide/session-policies-using.md",
      "user-guide/session-policies-managing.md", "user-guide/authentication-policies.md"], "tagging"),

    # ── Alerts ──
    ("creating and managing alerts in snowflake",
     ["user-guide/alerts.md", "user-guide/alerts-ui.md"], "alerts"),

    # ── Collaboration / Listings ──
    ("creating organizational listings in snowflake",
     ["user-guide/collaboration/listings/organizational/org-listing-create.md",
      "user-guide/collaboration/listings/organizational/org-listing-configure.md",
      "user-guide/collaboration/listings/organizational/org-listing-manage.md"], "collaboration"),
    ("organizational listing governance and auto fulfillment",
     ["user-guide/collaboration/listings/organizational/org-listing-governance.md",
      "user-guide/collaboration/listings/organizational/org-listing-auto-fulfillment.md"], "collaboration"),
    ("pricing plans and offers for snowflake marketplace",
     ["user-guide/collaboration/listings/pricing-plans-offers/pricing-plans-and-offers.md",
      "user-guide/collaboration/listings/pricing-plans-offers/providers-create-manage-pricing-plans.md",
      "user-guide/collaboration/listings/pricing-plans-offers/providers-create-manage-offers.md"], "collaboration"),
    ("organization profiles overview and management",
     ["user-guide/collaboration/organization-profiles/org-profiles-create-manage.md",
      "user-guide/collaboration/organization-profiles/org-profile-manifest-reference.md"], "collaboration"),

    # ── Spark Connector ──
    ("snowflake spark connector setup and usage",
     ["user-guide/spark-connector.md", "user-guide/spark-connector-install.md",
      "user-guide/spark-connector-overview.md", "user-guide/spark-connector-use.md"], "spark"),

    # ── JDBC/ODBC Drivers ──
    ("snowflake jdbc driver configuration and usage",
     ["developer-guide/jdbc/jdbc-download.md", "developer-guide/jdbc/jdbc-api.md"], "drivers"),
    ("snowflake odbc driver setup on windows mac linux",
     ["developer-guide/odbc/odbc-windows.md", "developer-guide/odbc/odbc-mac.md",
      "developer-guide/odbc/odbc-linux.md"], "drivers"),
    ("snowflake nodejs driver for application development",
     ["developer-guide/node-js/nodejs-driver.md", "developer-guide/node-js/nodejs-driver-install.md",
      "developer-guide/node-js/nodejs-driver-consume.md"], "drivers"),
    ("snowflake dotnet driver for .net applications",
     ["developer-guide/dotnet/dotnet-driver.md"], "drivers"),
    ("snowflake golang driver setup and queries",
     ["developer-guide/golang/go-driver.md"], "drivers"),

    # ── Snowflake CLI ──
    ("snowflake cli installation and command reference",
     ["developer-guide/snowflake-cli/index.md"], "snowflake-cli"),

    # ── Builders / Apps ──
    ("building snowflake native apps with builders framework",
     ["developer-guide/builders/devops.md", "developer-guide/builders/observability.md"], "apps"),
    ("snowflake application runtime and packaging",
     ["developer-guide/snowflake-app-runtime/about-snowflake-app-runtime.md",
      "developer-guide/snowflake-app-runtime/app-yml.md",
      "developer-guide/snowflake-app-runtime/limitations.md"], "apps"),

    # ── SQL API ──
    ("snowflake sql api for rest based queries",
     ["developer-guide/sql-api/index.md"], "sql-api"),

    # ── Unstructured Data ──
    ("working with unstructured data in snowflake",
     ["user-guide/unstructured-intro.md", "user-guide/unstructured-data-sharing.md",
      "user-guide/unstructured-data-java.md", "user-guide/unstructured-ts.md"], "unstructured"),

    # ── connectors ──
    ("informatica cloud connector for snowflake",
     ["connectors/informatica-cloud-connector.md"], "connectors"),

    # ── Tutorials ──
    ("getting started snowflake in 20 minutes",
     ["user-guide/tutorials/snowflake-in-20minutes.md"], "tutorials"),
    ("json data handling tutorial",
     ["user-guide/tutorials/json-basics-tutorial.md", "user-guide/semi-structured-tutorials.md"], "tutorials"),
    ("hybrid tables getting started tutorial",
     ["user-guide/tutorials/getting-started-with-hybrid-tables-tutorial.md",
      "user-guide/tables-hybrid-create.md"], "tutorials"),
    ("data quality tutorial",
     ["user-guide/tutorials/data-quality-tutorial-start.md",
      "user-guide/data-quality-intro.md"], "tutorials"),
    ("sensitive data auto classification tutorial",
     ["user-guide/tutorials/sensitive-data-auto-classification.md",
      "user-guide/classify-auto.md"], "tutorials"),
    ("bulk data loading tutorial from external stage",
     ["user-guide/tutorials/data-load-external-tutorial.md"], "tutorials"),

    # ── Diff Privacy ──
    ("differential privacy in snowflake",
     ["user-guide/diff-privacy.md", "user-guide/tutorials/diff-privacy.md",
      "sql-reference/functions-differential-privacy.md"], "compliance"),

    # ── Compliance ──
    ("compliance certifications overview",
     ["user-guide/intro-compliance.md",
      "user-guide/cert-iso-27001.md", "user-guide/cert-soc-2.md",
      "user-guide/cert-pci-dss.md", "user-guide/cert-fedramp.md"], "compliance"),
    ("hipaa and healthcare compliance",
     ["user-guide/cert-hitrust.md", "user-guide/cert-itar.md"], "compliance"),

    # ── Data Unloading ──
    ("unloading data from snowflake to s3",
     ["user-guide/data-unload-s3.md", "user-guide/data-unload-overview.md",
      "user-guide/data-unload-considerations.md"], "data-load"),
    ("unloading data to azure and gcs",
     ["user-guide/data-unload-azure.md", "user-guide/data-unload-gcs.md",
      "user-guide/data-unload-snowflake.md"], "data-load"),

    # ── Data Pipelines ──
    ("building data pipelines in snowflake",
     ["user-guide/data-pipelines-intro.md", "user-guide/data-pipelines-examples.md",
      "user-guide/streams-intro.md", "user-guide/tasks-intro.md"], "pipelines"),

    # ── Data Engineering ──
    ("dbt projects on snowflake best practices",
     ["user-guide/data-engineering/dbt-projects-on-snowflake-dependencies.md",
      "user-guide/data-engineering/dbt-projects-on-snowflake-versions.md",
      "user-guide/data-engineering/dbt-projects-on-snowflake-access-control.md"], "pipelines"),

    # ── Object Cloning and Dependencies ──
    ("cloning objects with create clone in snowflake",
     ["user-guide/object-clone.md", "user-guide/object-dependencies.md"], "objects"),

    # ── Data Loading ──
    ("loading data from s3 compatible storage",
     ["user-guide/data-load-s3-compatible-storage.md", "user-guide/data-load-s3-compatible-private.md"], "data-load"),
    ("loading data from azure with private endpoints",
     ["user-guide/data-load-azure-private.md", "user-guide/data-load-aws-private.md"], "data-load"),

    # ── Search Optimization ──
    ("search optimization service overview and enablement",
     ["user-guide/search-optimization-service.md",
      "user-guide/search-optimization/enabling.md",
      "user-guide/search-optimization/queries-that-benefit.md"], "search-opt"),
    ("search optimization monitoring and cost",
     ["user-guide/search-optimization/monitoring-search-optimization.md",
      "user-guide/search-optimization/cost-estimation.md",
      "user-guide/search-optimization/working-with-tables.md"], "search-opt"),
]


def generate_from_topics(existing_queries: set[str]) -> list[dict]:
    """Generate golden entries from hand-curated TOPIC_QUERIES."""
    entries = []
    for query, file_rel_paths, _topic in TOPIC_QUERIES:
        if query in existing_queries:
            continue

        # Verify files exist
        abs_files = [MARKDOWN_ROOT / p for p in file_rel_paths]
        existing = [p for p in abs_files if p.is_file()]
        if not existing:
            print(f"  WARNING: no files exist for query '{query[:60]}...', skipping")
            continue

        entry = {
            "query": query,
            "expected_files": [str(p.relative_to(MARKDOWN_ROOT)) for p in existing],
            "expected_publications": ["markdown"],
        }
        entries.append(entry)
    return entries


def main() -> None:
    existing = load_existing(GOLDEN_PATH)
    print(f"Existing golden set: {len(existing)} queries")

    entries = generate_from_topics(existing)
    print(f"New entries to add: {len(entries)}")

    # Backup original
    if GOLDEN_PATH.is_file():
        backup = GOLDEN_PATH.with_suffix(".jsonl.bak")
        GOLDEN_PATH.rename(backup)
        print(f"Backed up original to {backup}")

    # Write all entries (preserve original + new)
    all_entries = []
    if GOLDEN_PATH.is_file():
        with GOLDEN_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    # Re-add any from backup
    backup_path = GOLDEN_PATH.with_suffix(".jsonl.bak")
    if backup_path.is_file():
        with backup_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    candidate = json.loads(line)
                    if candidate["query"] not in {e["query"] for e in all_entries}:
                        all_entries.append(candidate)

    all_entries.extend(entries)

    # De-duplicate by query
    seen = set()
    deduped = []
    for e in all_entries:
        if e["query"] not in seen:
            seen.add(e["query"])
            deduped.append(e)

    # Write final
    with GOLDEN_PATH.open("w") as f:
        for entry in deduped:
            f.write(json.dumps(entry) + "\n")

    print(f"\nFinal golden set: {len(deduped)} queries (target: {TARGET})")
    if len(deduped) < TARGET:
        print(f"  WARNING: {TARGET - len(deduped)} queries short of target")

    # Quick summary
    topics = set()
    for e in entries:
        for f in e["expected_files"]:
            topics.add(f.split("/")[0])
    print(f"Topics covered: {sorted(topics)}")


if __name__ == "__main__":
    main()
