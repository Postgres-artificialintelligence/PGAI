CREATE OR REPLACE FUNCTION add_pseudo_column()
RETURNS TABLE (
    id INT,
    date DATE,
    name VARCHAR,
    value DOUBLE PRECISION,
    description TEXT,
    pseudo_column DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT sample_data.id, sample_data.date, sample_data.name, sample_data.value, sample_data.description, NULL::double precision AS pseudo_column
    FROM sample_data;
END;
$$ LANGUAGE plpgsql;


