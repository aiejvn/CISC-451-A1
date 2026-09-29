from .schema import schema_text

TEMPLATE = (
    "Given the following SQLite database schema, write a SQL query that answers "
    "the question.\n\n{schema}\n\nQuestion: {question}\n\n"
    "Respond with only the SQL query in a ```sql code block."
)


def build_user_prompt(question: str, db_id: str) -> str:
    return TEMPLATE.format(schema=schema_text(db_id), question=question)
