-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create enum for report types
create type report_type as enum ('10_Q', '10_K');

-- 3. Create the documents table (stores complete markdown)
create table if not exists documents (
    id bigserial primary key,
    filename text not null,
    markdown_content text not null,
    ticker text not null,
    fiscal_year int not null,
    fiscal_quarter int,  -- 1-4 for 10-Q, NULL for 10-K
    report_type report_type not null,
    created_at timestamptz default now(),
    
    -- Constraints
    constraint valid_document_quarter check (
        (report_type = '10_Q' and fiscal_quarter between 1 and 4) or
        (report_type = '10_K' and fiscal_quarter is null)
    )
);

-- 4. Create the chunks table (stores embedded chunks)
create table if not exists chunks (
    id bigserial primary key,
    document_id bigint not null references documents(id) on delete cascade,
    content text not null,
    metadata jsonb,
    embedding vector(1024),
    fts tsvector generated always as (to_tsvector('english', content)) stored
);

-- 5. Create Indexes for speed
create index on documents (ticker);
create index on documents (ticker, fiscal_year, fiscal_quarter);
create index on documents (report_type);

create index on chunks (document_id);
create index on chunks using hnsw (embedding vector_cosine_ops);
create index on chunks using gin (fts);
create index on chunks using gin (metadata);

-- 6. Hybrid Search Function with optional filters
create or replace function hybrid_search(
  query_text text,
  query_embedding vector(1024),
  match_count int,
  full_text_weight float default 1.0,
  semantic_weight float default 1.0,
  rrf_k int default 60,
  filter_ticker text default null,
  filter_fiscal_year int default null,
  filter_fiscal_quarter int default null,
  filter_report_type report_type default null
)
returns table (
  id bigint,
  document_id bigint,
  content text,
  metadata jsonb,
  filename text,
  ticker text,
  fiscal_year int,
  fiscal_quarter int,
  report_type report_type,
  score float
)
language plpgsql
as $$
begin
  return query
  with full_text as (
    select 
      c.id, 
      row_number() over(order by ts_rank_cd(c.fts, plainto_tsquery('english', query_text)) desc) as rank_ix
    from chunks c
    join documents d on c.document_id = d.id
    where c.fts @@ plainto_tsquery('english', query_text)
      and (filter_ticker is null or d.ticker = filter_ticker)
      and (filter_fiscal_year is null or d.fiscal_year = filter_fiscal_year)
      and (filter_fiscal_quarter is null or d.fiscal_quarter = filter_fiscal_quarter)
      and (filter_report_type is null or d.report_type = filter_report_type)
    limit match_count * 2
  ),
  semantic as (
    select 
      c.id, 
      row_number() over(order by c.embedding <=> query_embedding) as rank_ix
    from chunks c
    join documents d on c.document_id = d.id
    where (filter_ticker is null or d.ticker = filter_ticker)
      and (filter_fiscal_year is null or d.fiscal_year = filter_fiscal_year)
      and (filter_fiscal_quarter is null or d.fiscal_quarter = filter_fiscal_quarter)
      and (filter_report_type is null or d.report_type = filter_report_type)
    order by c.embedding <=> query_embedding
    limit match_count * 2
  )
  select
    c.id,
    c.document_id,
    c.content,
    c.metadata,
    d.filename,
    d.ticker,
    d.fiscal_year,
    d.fiscal_quarter,
    d.report_type,
    (
        (coalesce(1.0 / (rrf_k + f.rank_ix), 0.0) * full_text_weight) +
        (coalesce(1.0 / (rrf_k + s.rank_ix), 0.0) * semantic_weight)
    ) as score
  from chunks c
  join documents d on c.document_id = d.id
  left join full_text f on c.id = f.id
  left join semantic s on c.id = s.id
  where f.id is not null or s.id is not null
  order by score desc
  limit match_count;
end;
$$;

-- Enable RLS (Row Level Security)
alter table documents enable row level security;
alter table chunks enable row level security;
