from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/data_clean_all.csv", sep=",")


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    content = f"""
        career: {row['career']}\n
        total_students: {row['total_students']}\n
        new_graduates_number: {row['new_graduates_number']}\n
        public_cost: {row['public_cost']}\n
        private_cost: {row['private_cost']}\n
        public_quality_rating: {row['public_quality_rating']}\n
        private_quality_rating: {row['private_quality_rating']}\n
        occupation_rate: {row['occupation_rate']}\n
        unemployment_rate: {row['unemployment_rate']}\n
        informality_rate: {row['informality_rate']}\n
        quality_employment_probability: {row['quality_employment_probability']}\n
        average_salary: {row['average_salary']}\n
        career_rank: {row['career_rank']}\n
        women_salary: {row['women_salary']}\n
        men_salary: {row['men_salary']}\n
        under_30_salary: {row['under_30_salary']}\n
        over_30_salary: {row['over_30_salary']}\n
        formal_salary: {row['formal_salary']}\n
        informal_salary: {row['informal_salary']}\n
        postgrad_percentage: {row['postgrad_percentage']}\n
        postgrad_salary: {row['postgrad_salary']}\n
        salary_increase: {row['salary_increase']}\n
    """
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "career": row["career"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)
