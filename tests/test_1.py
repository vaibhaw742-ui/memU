from memu.database import build_database
from memu.config.settings import DatabaseConfig
from pydantic import BaseModel

# Define user scope
class UserScope(BaseModel):
    user_id: str
    workspace_id: str

# Configure database
config = DatabaseConfig(
    metadata_store={"provider": "postgres", "dsn": "postgresql+psycopg://postgres:postgres@postgres:5432/memu"},
    vector_index={"provider": "pgvector"}
)

# Build database
db = build_database(config=config, user_model=UserScope)

list_items = db.memory_item_repo.list_items()
# Create memory
# item = db.memory_item_repo.create_item(
#     resource_id=None,
#     memory_type="profile",
#     summary="User prefers dark mode",
#     embedding=embedder.embed("User prefers dark mode"),
#     user_data={"user_id": "alice", "workspace_id": "ws_1"},
#     reinforce=True
# )

# # Search memories
# results = db.memory_item_repo.vector_search_items(
#     query_vec=embedder.embed("What are the user's UI preferences?"),
#     top_k=5,
#     where={"user_id": "alice", "workspace_id": "ws_1"},
#     ranking="salience"
# )
print(list_items)
# Clean up
db.close()