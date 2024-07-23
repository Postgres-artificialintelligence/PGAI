#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/heapam.h"
#include "utils/rel.h"
#include "utils/tuplestore.h"
#include "utils/snapmgr.h"
#include "catalog/pg_class.h"
#include "nodes/makefuncs.h"
#include "utils/snapshot.h"
#include "utils/builtins.h"  // For text_to_cstring

PG_MODULE_MAGIC;

void _PG_init(void);
void _PG_fini(void);

PG_FUNCTION_INFO_V1(fetch_data_with_pseudo_column);

static Datum fetch_data_with_pseudo_column_internal(FunctionCallInfo fcinfo, const char *table_name);

void _PG_init(void)
{
    /* Initialization code goes here */
}

void _PG_fini(void)
{
    /* Finalization code goes here */
}

Datum
fetch_data_with_pseudo_column(PG_FUNCTION_ARGS)
{
    text *table_name_text = PG_GETARG_TEXT_PP(0);
    char *table_name = text_to_cstring(table_name_text);
    return fetch_data_with_pseudo_column_internal(fcinfo, table_name);
}

static Datum
fetch_data_with_pseudo_column_internal(FunctionCallInfo fcinfo, const char *table_name)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate *tupstore;
    MemoryContext per_query_ctx;
    MemoryContext oldcontext;
    Relation rel;
    Snapshot snapshot;
    TableScanDesc scan;
    HeapTuple tuple;

    /* Check if caller supports returning a tuplestore */
    if (rsinfo == NULL)
        return (Datum) 0;
    /* Check if materialize mode is supported */
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("Materialize mode required, but it is not allowed in this context.")));
    /* Build a tuple descriptor for our result type */
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        elog(ERROR, "[pgai] fetch_data_with_pseudo_column_internal: Return type must be a row type.");

    int natts = tupdesc->natts;

    /* Create a new tuple descriptor with an additional column for the pseudo column */
    TupleDesc new_tupdesc = CreateTemplateTupleDesc(natts + 1);
    for (int i = 0; i < natts; i++)
    {
        TupleDescInitEntry(new_tupdesc, i + 1,
                           NameStr(tupdesc->attrs[i].attname),
                           tupdesc->attrs[i].atttypid,
                           tupdesc->attrs[i].atttypmod,
                           tupdesc->attrs[i].attndims);
    }
    TupleDescInitEntry(new_tupdesc, natts + 1, "pseudo_column", INT4OID, -1, 0);

    /* Switch into long-lived context to construct returned data structures */
    per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    oldcontext = MemoryContextSwitchTo(per_query_ctx);

    /* Initialize tuplestore */
    tupstore = tuplestore_begin_heap(true, false, 1000);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = new_tupdesc;
    MemoryContextSwitchTo(oldcontext);

    /* Open the relation */
    rel = relation_openrv(makeRangeVar("public", (char *)table_name, -1), AccessShareLock);
    if (!RelationIsValid(rel))
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("Relation not found")));

    /* Begin table scan */
    snapshot = GetTransactionSnapshot();
    scan = table_beginscan(rel, snapshot, 0, NULL);

    /* Allocate memory for values and nulls arrays */
    Datum *values = (Datum *) palloc((natts + 1) * sizeof(Datum));
    bool *nulls = (bool *) palloc((natts + 1) * sizeof(bool));

    /* Fetch rows and add to tuplestore */
    while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL)
    {
        /* Extract values for existing columns */
        for (int i = 0; i < natts; i++)
        {
            values[i] = heap_getattr(tuple, i + 1, tupdesc, &nulls[i]);
        }

        /* Add pseudo column value */
        values[natts] = Int32GetDatum(1000);  // Example value for the pseudo column
        nulls[natts] = false;

        tuplestore_putvalues(tupstore, new_tupdesc, values, nulls);
    }

    /* Cleanup */
    table_endscan(scan);
    relation_close(rel, AccessShareLock);
    tuplestore_donestoring(tupstore);

    /* Free allocated memory */
    pfree(values);
    pfree(nulls);

    return (Datum) 0;
}
