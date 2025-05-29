from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import json
import logging
from typing import List, Dict
from pydantic import BaseModel
import aiofiles
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    disclaimer: str

app = FastAPI(title="Medical Chat Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("vector_stores")
DOCUMENTS_FILE = "documents.json"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Global variables
embed_model = None
qa_model = None
documents = []
document_metadata = []
uploaded_files = []  # Track uploaded files

@app.on_event("startup")
async def startup_event():
    """Initialize models and load existing data"""
    global embed_model, qa_model, documents, document_metadata
    
    try:
        logger.info("Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("Loading QA model...")
        # Using a more robust QA model
        qa_model = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        
        # Load existing documents
        await load_documents()
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Fallback to original model if the new one fails
        try:
            qa_model = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
            logger.info("Loaded fallback QA model")
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {fallback_error}")
            # Use a basic model as last resort
            qa_model = pipeline("question-answering")
            logger.info("Loaded basic QA model")

async def load_documents():
    """Load documents from JSON file"""
    global documents, document_metadata, uploaded_files
    
    docs_file = VECTOR_STORE_DIR / DOCUMENTS_FILE
    if docs_file.exists():
        try:
            async with aiofiles.open(docs_file, 'r', encoding='utf-8') as f:
                data = json.loads(await f.read())
                documents = data.get('documents', [])
                document_metadata = data.get('metadata', [])
                uploaded_files = data.get('uploaded_files', [])
                logger.info(f"Loaded {len(documents)} documents from {len(uploaded_files)} files")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            documents = []
            document_metadata = []
            uploaded_files = []

async def save_documents():
    """Save documents to JSON file"""
    docs_file = VECTOR_STORE_DIR / DOCUMENTS_FILE
    try:
        data = {
            'documents': documents,
            'metadata': document_metadata,
            'uploaded_files': uploaded_files
        }
        async with aiofiles.open(docs_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))
        logger.info("Documents saved successfully")
    except Exception as e:
        logger.error(f"Error saving documents: {e}")

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove page numbers, headers, footers (basic patterns)
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\d+\s*$', '', text)  # Numbers at end of lines
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\s]', '', text)
    return text

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with sentence boundaries"""
    if not text or len(text.strip()) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Clean the text first
    text = clean_text(text)
    if not text:
        return []
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_words = words[-overlap//5:] if len(words) > overlap//5 else words
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    try:
        validate_file(file)
        
        # Check if file already uploaded
        if file.filename in [f['filename'] for f in uploaded_files]:
            return {
                "status": "info",
                "message": f"Document '{file.filename}' was already uploaded previously.",
                "chunks_processed": 0
            }
        
        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Save file temporarily
        file_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        
        # Process document
        full_text = ""
        try:
            if file.filename.lower().endswith('.pdf'):
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    if page_text.strip():  # Only add non-empty pages
                        full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                doc.close()
            else:  # txt file
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = await f.read()
        except Exception as e:
            logger.error(f"Error reading file content: {e}")
            raise HTTPException(status_code=400, detail=f"Could not read file content: {str(e)}")
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or unreadable")
        
        # Clean and create chunks
        full_text = clean_text(full_text)
        chunks = chunk_text(full_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from file")
        
        logger.info(f"Created {len(chunks)} chunks from document '{file.filename}'")
        
        # Generate embeddings
        new_vectors = []
        chunks_added = 0
        
        for i, chunk in enumerate(chunks):
            if chunk.strip() and len(chunk.strip()) > 20:  # Skip very short chunks
                documents.append(chunk)
                document_metadata.append({
                    'filename': file.filename,
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'file_type': Path(file.filename).suffix.lower()
                })
                
                try:
                    vector = embed_model.encode(chunk)
                    new_vectors.append(vector)
                    chunks_added += 1
                except Exception as e:
                    logger.error(f"Error encoding chunk {i}: {e}")
                    # Remove the chunk and metadata if encoding failed
                    documents.pop()
                    document_metadata.pop()
        
        if not new_vectors:
            raise HTTPException(status_code=400, detail="Could not process any chunks from the document")
        
        # Update FAISS index
        try:
            vectors_array = np.array(new_vectors).astype('float32')
            
            index_path = VECTOR_STORE_DIR / "vector_store.faiss"
            if index_path.exists():
                # Load existing index and add new vectors
                index = faiss.read_index(str(index_path))
                index.add(vectors_array)
            else:
                # Create new index
                index = faiss.IndexFlatL2(vectors_array.shape[1])
                index.add(vectors_array)
            
            faiss.write_index(index, str(index_path))
            logger.info(f"Updated FAISS index with {len(new_vectors)} vectors")
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
            raise HTTPException(status_code=500, detail="Error updating search index")
        
        # Add file to uploaded files list
        uploaded_files.append({
            'filename': file.filename,
            'upload_time': str(Path(file_path).stat().st_mtime),
            'chunks_count': chunks_added,
            'file_size': len(contents)
        })
        
        # Save documents
        await save_documents()
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully. Added {chunks_added} text chunks.",
            "chunks_processed": chunks_added,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

def get_relevant_contexts(question: str, top_k: int = 5) -> List[str]:
    """Get relevant contexts using vector search"""
    try:
        if not documents:
            logger.warning("No documents available for search")
            return []
            
        index_path = VECTOR_STORE_DIR / "vector_store.faiss"
        if not index_path.exists():
            logger.warning("Vector store index not found")
            return []
        
        index = faiss.read_index(str(index_path))
        if index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
            
        q_vector = embed_model.encode(question).reshape(1, -1).astype('float32')
        
        # Search for top matches
        k = min(top_k, len(documents), index.ntotal)
        distances, indices = index.search(q_vector, k=k)
        
        logger.info(f"Search results - distances: {distances[0][:3]}, indices: {indices[0][:3]}")
        
        # Collect relevant contexts
        contexts = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(documents) and idx >= 0:
                # Use adaptive threshold - take top results and those below distance threshold
                if i < 3 or distance < 1.5:  # Always take top 3, plus others below threshold
                    contexts.append(documents[idx])
                    logger.info(f"Added context {i}: distance={distance:.3f}")
                
        return contexts[:top_k]  # Limit to top_k contexts
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []

def generate_answer_from_context(question: str, contexts: List[str]) -> tuple:
    """Generate answer using multiple contexts"""
    if not contexts:
        return "", 0.0
    
    best_answer = ""
    best_confidence = 0.0
    
    # Try each context individually first
    for i, context in enumerate(contexts):
        try:
            # Truncate context to fit model limits
            max_context_length = 1500  # Leave room for question
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            result = qa_model(question=question, context=context)
            answer = result.get('answer', '').strip()
            confidence = result.get('score', 0.0)
            
            logger.info(f"Context {i}: answer='{answer[:50]}...', confidence={confidence:.3f}")
            
            if confidence > best_confidence and len(answer) > 3:
                best_answer = answer
                best_confidence = confidence
                
        except Exception as e:
            logger.error(f"Error processing context {i}: {e}")
            continue
    
    # If no good answer from individual contexts, try combined context
    if best_confidence < 0.3 and len(contexts) > 1:
        try:
            combined_context = " ".join(contexts[:3])  # Use top 3 contexts
            if len(combined_context) > 1500:
                combined_context = combined_context[:1500] + "..."
            
            result = qa_model(question=question, context=combined_context)
            combined_answer = result.get('answer', '').strip()
            combined_confidence = result.get('score', 0.0)
            
            logger.info(f"Combined context: answer='{combined_answer[:50]}...', confidence={combined_confidence:.3f}")
            
            if combined_confidence > best_confidence:
                best_answer = combined_answer
                best_confidence = combined_confidence
                
        except Exception as e:
            logger.error(f"Error with combined context: {e}")
    
    return best_answer, best_confidence

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer questions based on uploaded documents"""
    try:
        question = request.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Received question: {question}")
        
        # Check if we have documents
        if not documents:
            return ChatResponse(
                answer="Please upload a medical document first before asking questions. Use the upload section above to add PDF or TXT files.",
                confidence=0.0,
                disclaimer="⚠️ This is an AI assistant. Always consult healthcare professionals for medical advice."
            )
        
        # Default disclaimer
        disclaimer = "⚠️ This is an AI assistant. Always consult healthcare professionals for medical advice."
        
        # Get relevant contexts
        contexts = get_relevant_contexts(question, top_k=5)
        
        if not contexts:
            return ChatResponse(
                answer="I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or ensure the document contains the information you're looking for.",
                confidence=0.0,
                disclaimer=disclaimer
            )
        
        # Generate answer
        answer, confidence = generate_answer_from_context(question, contexts)
        
        # Post-process answer
        if not answer or len(answer.strip()) < 3:
            # Provide a more informative fallback
            answer = f"I found some relevant information in the document about your question, but I cannot provide a specific answer. The document contains information related to '{question}'. Please review the document directly or ask a more specific question."
            confidence = 0.2
        elif len(answer) < 10 and confidence < 0.5:
            # For very short answers with low confidence, provide more context
            answer = f"Based on the document, the answer appears to be: {answer}. However, I recommend reviewing the full document for complete information about '{question}'."
        
        # Enhanced medical disclaimer for sensitive topics
        medical_terms = ['diagnose', 'diagnosis', 'treatment', 'medicine', 'drug', 'symptom', 'cure', 'therapy', 'disease', 'condition', 'medication', 'dosage', 'prescription']
        if any(term in question.lower() for term in medical_terms):
            disclaimer = "⚠️ IMPORTANT: This AI cannot provide medical diagnosis or treatment advice. Please consult a qualified healthcare professional for medical guidance."
        
        logger.info(f"Generated answer with confidence {confidence:.3f}")
        
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            disclaimer=disclaimer
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing your question")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_loaded": len(documents),
        "models_loaded": embed_model is not None and qa_model is not None,
        "uploaded_files": len(uploaded_files)
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_documents": len(documents),
        "total_chunks": len(documents),
        "uploaded_files_count": len(uploaded_files),
        "uploaded_files": [{"filename": f["filename"], "chunks": f["chunks_count"]} for f in uploaded_files],
        "vector_store_exists": (VECTOR_STORE_DIR / "vector_store.faiss").exists()
    }

@app.get("/uploaded-files")
async def get_uploaded_files():
    """Get list of uploaded files"""
    return {
        "files": uploaded_files,
        "total_count": len(uploaded_files)
    }

@app.get("/")
async def root():
    return {"message": "Medical Chat Assistant API is running", "status": "healthy"}

# Debug endpoints (remove in production)
@app.get("/debug/documents")
async def debug_documents():
    """Debug endpoint to check loaded documents"""
    return {
        "total_documents": len(documents),
        "sample_documents": [doc[:200] + "..." if len(doc) > 200 else doc for doc in documents[:3]],
        "metadata_sample": document_metadata[:3] if document_metadata else [],
        "uploaded_files": uploaded_files
    }

@app.get("/debug/search")
async def debug_search(q: str):
    """Debug endpoint to test search functionality"""
    contexts = get_relevant_contexts(q, top_k=3)
    return {
        "question": q,
        "contexts_found": len(contexts),
        "contexts": [ctx[:200] + "..." if len(ctx) > 200 else ctx for ctx in contexts]
    }