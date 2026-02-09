from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import settings

app = FastAPI(title="SharkGuard API", version="1.0")

@app.get("/")
async def root():
    return {"message": "Welcome to SharkGuard API"}

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)