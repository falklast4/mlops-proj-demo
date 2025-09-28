import logging

async def get_db():
    """Placeholder - no real database connection"""
    return None

async def save_prediction_log(**kwargs):
    """Placeholder - just log instead of saving to database"""
    logging.info(f"Would save prediction log: {kwargs}")
    pass