from sqlalchemy import create_engine
from sqlmodel import SQLModel

DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/smartAssistantDb"

def get_enginge():
    engine = create_engine(DATABASE_URL, echo=True)
    return engine

def create_tables(engine):
    SQLModel.metadata.create_all(engine)
