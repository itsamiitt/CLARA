# Legacy PostgreSQL migrations

These two `.sql` files are the never-wired PostgreSQL/pgvector migrations from the
original CLARA design (`docs/history/clara_implementation_plan.md`). No Python code
ever read or applied them. The live schema is created by SQLAlchemy `create_all`
plus the SQLite versioning module `clara/db/migrations.py`; these files are kept
only for historical reference.
