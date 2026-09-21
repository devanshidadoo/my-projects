from fastapi import FastAPI, HTTPException, Body, Depends, APIRouter
from typing import Optional, Any
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import os
import sys
from typing import Dict, List, Optional, Any
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import Config as config

from core.config import *
from utils import get_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Supported collections
messages = MESSAGE
user_inputs = USER_INPUTS
tasks = TASKS

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

MONGO_URI = config.MONGO_URI
MONGO_DB_NAME = config.MONGO_DB_NAME

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

@app.get("/api/data")
async def get_data(
    collection_name: str,
    _id: str,
):

    logger = get_logger(_id)
    logger.info(
        f"GET /api/data called with ID={_id} and collection_name={collection_name}"
    )

    # Validate collection name
    if collection_name not in [messages, user_inputs, tasks]:
        logger.error(f"Invalid collection_name={collection_name} for _id={_id}")
        raise HTTPException(status_code=400, detail="Invalid collection_name")

    try:

        obj_id = ObjectId(_id)

        collection = db[collection_name]
        result = collection.find_one({"_id": obj_id})

        if result is None:
            logger.error(f"Error fetching data for _id={_id}: Resource not found.")
            raise HTTPException(status_code=404, detail="Resource not found")

        # Convert ObjectId to string for response
        result["_id"] = str(result["_id"])

        processed_result = result

        logger.info(f"Successfully fetched data of {collection_name} for _id={_id}")
        return {
            "status": "success",
            "data": processed_result,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating data for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/api/data")
async def post_data(collection_name: str, _id: str, data: dict = Body(...)):

    logger = get_logger(_id)
    logger.info(
        f"POST /api/data called with collection_name={collection_name} and _id={_id}"
    )

    # Validate collection name
    valid_collections = [messages, user_inputs, tasks]
    if collection_name not in valid_collections:
        logger.error(f"Invalid collection_name={collection_name} for _id={_id}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection_name. Must be one of {valid_collections}",
        )

    try:
        # Convert _id to ObjectId
        obj_id = ObjectId(_id)
        collection = db[collection_name]

        # Check if the document with the given _id exists
        existing_doc = collection.find_one({"_id": obj_id})

        if existing_doc:
            # Replace the entire document keeping the _id
            data["_id"] = obj_id
            result = collection.replace_one({"_id": obj_id}, data)

            if result.modified_count == 0:
                logger.error(
                    f"Failed to update document in {collection_name} for _id={_id}"
                )
                raise HTTPException(status_code=500, detail="Failed to update document")
        else:
            # Insert the document with the provided data and _id
            doc_to_insert = data
            doc_to_insert["_id"] = obj_id

            result = collection.insert_one(doc_to_insert)

            if not result.acknowledged:
                logger.error(
                    f"Failed to insert document in {collection_name} for _id={_id}"
                )
                raise HTTPException(status_code=500, detail="Failed to insert document")

        return {"status": "success", "timestamp": datetime.now().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error posting data for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.put("/api/data")
async def put_data(
    collection_name: str,
    _id: str,
    key: str,
    task_id: Optional[str] = None,
    data: Any = Body(...),
):
    logger = get_logger(_id)
    logger.info(
        f"PUT /api/data called with collection_name={collection_name}, _id={_id}, key={key}, task_id={task_id}"
    )

    # Validate collection name
    valid_collections = [user_inputs, tasks]
    if collection_name not in valid_collections:
        logger.error(f"Invalid collection_name={collection_name} for _id={_id}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection_name. Must be one of {valid_collections}",
        )
    
    # Validate task_id is provided when updating tasks collection
    if collection_name == tasks and not task_id:
        logger.error("task_id is required when updating tasks collection")
        raise HTTPException(
            status_code=400,
            detail="task_id is required when updating tasks collection"
        )

    try:
        # Convert _id to ObjectId
        obj_id = ObjectId(_id)
        collection = db[collection_name]

        # Check if the document with the given _id exists
        existing_doc = collection.find_one({"_id": obj_id})

        if not existing_doc:
            logger.error(f"Document with _id={_id} not found in {collection_name}")
            raise HTTPException(
                status_code=404, detail=f"Document with _id {_id} not found"
            )

        if collection_name == user_inputs:
            # Define valid keys for user_inputs
            valid_user_input_keys = [
                "user_inputs",  # For updating the whole object
                "name",
                "language",
                "location",
                "land_size", 
                "soil_type",
                "water_source",
                "budget",
                "experience_level",
                "crop_preferences",
                "current_crops",
                "farming_season",
                "challenges",
                "goals",
            ]

            # Validate key
            if key not in valid_user_input_keys:
                logger.error(
                    f"Invalid key={key} for user_inputs in collection {collection_name}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid key for user_inputs. Must be one of {valid_user_input_keys}",
                )

            # If key is "user_inputs", replace the entire user_inputs object
            if key == "user_inputs":
                # Validate that all keys in the data are allowed
                if isinstance(data, dict):
                    invalid_keys = [
                        k for k in data.keys() if k not in valid_user_input_keys[1:]
                    ]
                    if invalid_keys:
                        logger.error(f"Invalid keys in user_inputs: {invalid_keys}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid keys in user_inputs: {invalid_keys}. Valid keys are: {valid_user_input_keys[1:]}",
                        )
                existing_doc["user_inputs"] = data
            else:
                # Check if user_inputs exists, create if not
                if "user_inputs" not in existing_doc:
                    existing_doc["user_inputs"] = {}

                # Update specific field within user_inputs
                existing_doc["user_inputs"][key] = data

            # Replace the entire document with the updated version
            result = collection.replace_one({"_id": obj_id}, existing_doc)

            if result.matched_count == 0:
                logger.error(
                    f"Failed to update document in {collection_name} for _id={_id}"
                )
                raise HTTPException(status_code=500, detail="Failed to update document")

            return {"status": "success", "timestamp": datetime.now().isoformat()}

        elif collection_name == tasks:
            # For tasks, expect key to be "tasks" and task_id to identify the specific task
            if key != "tasks":
                logger.error(f"For tasks collection, key must be 'tasks', got {key}")
                raise HTTPException(
                    status_code=400,
                    detail="For tasks collection, key must be 'tasks'"
                )
            
            if not task_id:
                logger.error("task_id is required when updating tasks")
                raise HTTPException(
                    status_code=400,
                    detail="task_id is required when updating tasks"
                )
            
            if not isinstance(data, str):
                logger.error(f"Status update for task must be a string, got {type(data)}")
                raise HTTPException(
                    status_code=400,
                    detail="Status update for task must be a string."
                )
            
            if "tasks" not in existing_doc or not isinstance(existing_doc["tasks"], list):
                logger.error(f"No tasks array found in document with _id={_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"No tasks array found in document with _id={_id}"
                )
            
            # Find the task with the given task_id
            task_found = False
            for task in existing_doc["tasks"]:
                if task.get("task_id") == task_id:
                    task["status"] = data
                    task_found = True
                    break
            
            if not task_found:
                logger.error(f"Task with task_id={task_id} not found in document with _id={_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Task with task_id={task_id} not found in document with _id={_id}"
                )
            
            # Replace the entire document with the updated version
            result = collection.replace_one({"_id": obj_id}, existing_doc)

            if result.matched_count == 0:
                logger.error(
                    f"Failed to update task in {collection_name} for _id={_id}"
                )
                raise HTTPException(status_code=500, detail="Failed to update task status")

            return {"status": "success", "timestamp": datetime.now().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating data for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/data")
async def delete_data(collection_name: str, _id: str):
    logger = get_logger(_id)
    logger.info(
        f"DELETE /api/data called with collection_name={collection_name} and _id={_id}"
    )

    # Validate collection name
    valid_collections = [messages, user_inputs, tasks]
    if collection_name not in valid_collections:
        logger.error(f"Invalid collection_name={collection_name} for _id={_id}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection_name. Must be one of {valid_collections}",
        )

    try:
        # Convert _id to ObjectId
        obj_id = ObjectId(_id)
        collection = db[collection_name]

        # Check if the document with the given _id exists
        existing_doc = collection.find_one({"_id": obj_id})

        if not existing_doc:
            logger.error(f"Document with _id={_id} not found in {collection_name}")
            raise HTTPException(
                status_code=404, detail=f"Document with _id {_id} not found"
            )

        # Delete the document
        result = collection.delete_one({"_id": obj_id})

        if result.deleted_count == 0:
            logger.error(
                f"Failed to delete document in {collection_name} for _id={_id}"
            )
            raise HTTPException(status_code=500, detail="Failed to delete document")

        logger.info(f"Successfully deleted document from {collection_name} for _id={_id}")
        return {"status": "success", "timestamp": datetime.now().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/task")
async def add_task(
    _id: str,
    task_data: dict = Body(...)
):
    """
    Add a new task to the tasks array of a document.
    Expected task_data format: {"task_id": "...", "content": "..."}
    """
    logger = get_logger(_id)
    logger.info(f"POST /api/task called with _id={_id}")

    try:
        # Convert _id to ObjectId
        obj_id = ObjectId(_id)
        collection = db[tasks]

        # Validate required fields in task_data
        required_fields = ["task_id", "content"]
        for field in required_fields:
            if field not in task_data:
                logger.error(f"Missing required field: {field}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        # Check if the document exists
        existing_doc = collection.find_one({"_id": obj_id})

        if existing_doc:
            # Initialize tasks array if it doesn't exist
            if "tasks" not in existing_doc:
                existing_doc["tasks"] = []
            
            # Check if task_id already exists
            for task in existing_doc["tasks"]:
                if task.get("task_id") == task_data["task_id"]:
                    logger.error(f"Task with task_id={task_data['task_id']} already exists")
                    raise HTTPException(
                        status_code=409,
                        detail=f"Task with task_id={task_data['task_id']} already exists"
                    )
            
            # Add the new task
            existing_doc["tasks"].append(task_data)
            
            # Update the document
            result = collection.replace_one({"_id": obj_id}, existing_doc)
            
            if result.matched_count == 0:
                logger.error(f"Failed to update document for _id={_id}")
                raise HTTPException(status_code=500, detail="Failed to update document")
        else:
            # Create new document with the task
            new_doc = {
                "_id": obj_id,
                "tasks": [task_data]
            }
            result = collection.insert_one(new_doc)
            
            if not result.acknowledged:
                logger.error(f"Failed to create document for _id={_id}")
                raise HTTPException(status_code=500, detail="Failed to create document")

        logger.info(f"Successfully added task with task_id={task_data['task_id']} for _id={_id}")
        return {"status": "success", "timestamp": datetime.now().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding task for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.delete("/api/task")
async def delete_task(
    _id: str,
    task_id: str
):
    """
    Delete a specific task from the tasks array of a document.
    """
    logger = get_logger(_id)
    logger.info(f"DELETE /api/task called with _id={_id}, task_id={task_id}")

    try:
        # Convert _id to ObjectId
        obj_id = ObjectId(_id)
        collection = db[tasks]

        # Check if the document exists
        existing_doc = collection.find_one({"_id": obj_id})

        if not existing_doc:
            logger.error(f"Document with _id={_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Document with _id={_id} not found"
            )

        if "tasks" not in existing_doc or not isinstance(existing_doc["tasks"], list):
            logger.error(f"No tasks array found in document with _id={_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No tasks array found in document with _id={_id}"
            )

        # Find and remove the task
        original_length = len(existing_doc["tasks"])
        existing_doc["tasks"] = [
            task for task in existing_doc["tasks"] 
            if task.get("task_id") != task_id
        ]

        if len(existing_doc["tasks"]) == original_length:
            logger.error(f"Task with task_id={task_id} not found in document with _id={_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Task with task_id={task_id} not found"
            )

        # Update the document
        result = collection.replace_one({"_id": obj_id}, existing_doc)

        if result.matched_count == 0:
            logger.error(f"Failed to update document for _id={_id}")
            raise HTTPException(status_code=500, detail="Failed to update document")

        logger.info(f"Successfully deleted task with task_id={task_id} for _id={_id}")
        return {"status": "success", "timestamp": datetime.now().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/task")
async def get_tasks(
    _id: str,
    task_id: Optional[str] = None
):
    """
    Get tasks for a document. If task_id is provided, return specific task.
    Otherwise, return all tasks for the document.
    """
    logger = get_logger(_id)
    logger.info(f"GET /api/task called with _id={_id}, task_id={task_id}")

    try:
        # Convert _id to ObjectId
        obj_id = ObjectId(_id)
        collection = db[tasks]

        # Find the document
        existing_doc = collection.find_one({"_id": obj_id})

        if not existing_doc:
            logger.error(f"Document with _id={_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Document with _id={_id} not found"
            )

        if "tasks" not in existing_doc:
            existing_doc["tasks"] = []

        tasks_data = existing_doc["tasks"]

        if task_id:
            # Return specific task
            for task in tasks_data:
                if task.get("task_id") == task_id:
                    logger.info(f"Successfully retrieved task with task_id={task_id} for _id={_id}")
                    return {
                        "status": "success",
                        "data": task,
                        "timestamp": datetime.now().isoformat()
                    }
            
            logger.error(f"Task with task_id={task_id} not found in document with _id={_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Task with task_id={task_id} not found"
            )
        else:
            # Return all tasks
            logger.info(f"Successfully retrieved all tasks for _id={_id}")
            return {
                "status": "success",
                "data": tasks_data,
                "timestamp": datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving tasks for _id={_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}
