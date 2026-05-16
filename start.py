import os
import uvicorn

if __name__ == "__main__":
    # Read PORT from environment (Render sets this automatically)
    # Fall back to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Never use reload=True in production
    )