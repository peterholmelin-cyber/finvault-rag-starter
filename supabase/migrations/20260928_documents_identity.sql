-- supabase/migrations/20260928_documents_identity.sql
-- One document per (ticker, period, report_type); re-ingest = replace.
create unique index if not exists documents_identity_key
    on documents (
        ticker,
        fiscal_year,
        report_type,
        coalesce(fiscal_quarter, -1)
    );
