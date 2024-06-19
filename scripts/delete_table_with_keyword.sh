#!/bin/bash

# Default values
DEFAULT_DB_HOST="default_host"
DEFAULT_DB_PORT="default_port"
DEFAULT_PHRASE="default_phrase"
DEFAULT_DB_NAME="pipeline"
DEFAULT_DB_USER="postgres"
DEFAULT_DB_PASSWORD="pipe"

# Command line arguments
DB_HOST=${1:-$DEFAULT_DB_HOST}
DB_PORT=${2:-$DEFAULT_DB_PORT}
PHRASE=${3:-$DEFAULT_PHRASE}
DB_NAME=${4:-$DEFAULT_DB_NAME}
DB_USER=${5:-$DEFAULT_DB_USER}
DB_PASSWORD=${6:-$DEFAULT_DB_PASSWORD}


# Export the PGPASSWORD environment variable
export PGPASSWORD=$DB_PASSWORD

# Get the list of tables containing the phrase
tables=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "
SELECT tablename 
FROM pg_tables 
WHERE tablename LIKE '%$PHRASE%';")

# Loop through the list of tables and drop each one
for table in $tables; do
    if [[ ! -z "$table" ]]; then
        echo "Dropping table: $table"
        psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS $table CASCADE;"
    fi
done

# Unset the PGPASSWORD environment variable
unset PGPASSWORD

echo "All tables containing the phrase '$PHRASE' have been removed."