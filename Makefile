MODULES = pgai
EXTENSION = pgai
DATA = pgai--1.0.sql
PGFILEDESC = "pgai - example extension for PostgreSQL"

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

PG_CPPFLAGS = -I$(libdir)/postgresql/include
PG_CFLAGS = -std=c99 -fPIC
