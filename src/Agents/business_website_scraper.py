"""
Business Website Scraper Agent
An agentic workflow that scrapes business websites, generates Q&A content,
and populates the knowledge base.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentStep:
    """Represents a single step in the agent workflow."""
    name: str
    description: str
    status: AgentStatus = AgentStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


@dataclass
class AgentResult:
    """Result of an agent execution."""
    success: bool
    message: str
    steps: List[AgentStep]
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "steps": [s.to_dict() for s in self.steps],
            "data": self.data
        }


class BusinessWebsiteScraperAgent:
    """
    Agent that scrapes business websites and generates Q&A knowledge base content.
    
    Workflow:
    1. Validate inputs and project
    2. Scrape the target website
    3. Store scraped content in database
    4. Generate relevant prompts/questions
    5. Generate answers for each prompt
    6. Create embeddings and store Q&A pairs
    """
    
    AGENT_NAME = "Business Website Scraper"
    AGENT_VERSION = "1.0.0"
    
    def __init__(
        self,
        db_backend,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the Business Website Scraper Agent.
        
        Args:
            db_backend: Database backend instance
            progress_callback: Optional callback(step_name, current, total) for progress updates
        """
        self.db = db_backend
        self.progress_callback = progress_callback
        self.steps: List[AgentStep] = []
        self._cancelled = False
    
    def _add_step(self, name: str, description: str) -> AgentStep:
        """Add a new step to the workflow."""
        step = AgentStep(name=name, description=description)
        self.steps.append(step)
        return step
    
    def _update_step(
        self, 
        step: AgentStep, 
        status: AgentStatus, 
        result: Any = None,
        error: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Update a step's status and result."""
        step.status = status
        step.result = result
        step.error = error
        
        if status == AgentStatus.RUNNING:
            step.started_at = datetime.now()
        elif status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
            step.completed_at = datetime.now()
        
        if metadata:
            step.metadata.update(metadata)
    
    def _report_progress(self, step_name: str, current: int, total: int):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(step_name, current, total)
    
    def cancel(self):
        """Cancel the running agent."""
        self._cancelled = True
    
    def run(
        self,
        project_name: str,
        url: str,
        max_pages: int = 10,
        business_context: Optional[str] = None,
        default_weight: float = 1.0
    ) -> AgentResult:
        """
        Execute the Business Website Scraper workflow.
        
        Args:
            project_name: Name of the target project
            url: Starting URL to scrape
            max_pages: Maximum number of pages to scrape
            business_context: Optional context about the business
            default_weight: Default weight for generated Q&A pairs
            
        Returns:
            AgentResult with success status and generated data
        """
        self.steps = []
        self._cancelled = False
        
        # Initialize result data
        result_data = {
            "pages_scraped": 0,
            "prompts_generated": 0,
            "qa_pairs_added": 0,
            "domains": []
        }
        
        try:
            # ================================================================
            # STEP 1: Validate Project
            # ================================================================
            step1 = self._add_step(
                "validate_project",
                f"Validating project '{project_name}'"
            )
            self._update_step(step1, AgentStatus.RUNNING)
            self._report_progress("validate_project", 0, 6)
            
            metadata = self.db.get_project_metadata(project_name)
            if not metadata:
                self._update_step(step1, AgentStatus.FAILED, error=f"Project '{project_name}' not found")
                return AgentResult(
                    success=False,
                    message=f"Project '{project_name}' not found",
                    steps=self.steps
                )
            
            embedding_model = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self._update_step(step1, AgentStatus.COMPLETED, metadata={
                "embedding_model": embedding_model,
                "vector_dimension": metadata.get("vector_dimension")
            })
            
            if self._cancelled:
                return self._cancelled_result()
            
            # ================================================================
            # STEP 2: Scrape Website
            # ================================================================
            step2 = self._add_step(
                "scrape_website",
                f"Scraping website: {url} (max {max_pages} pages)"
            )
            self._update_step(step2, AgentStatus.RUNNING)
            self._report_progress("scrape_website", 1, 6)
            
            from ..Scraper.scraper_universal import scrape_website
            
            scraped_data = scrape_website(url, max_pages)
            
            if not scraped_data:
                self._update_step(step2, AgentStatus.FAILED, error="No content could be scraped from the URL")
                return AgentResult(
                    success=False,
                    message="No content could be scraped from the provided URL",
                    steps=self.steps
                )
            
            result_data["pages_scraped"] = len(scraped_data)
            result_data["domains"] = list(set(page.get("domain", "") for page in scraped_data))
            
            self._update_step(step2, AgentStatus.COMPLETED, metadata={
                "pages_scraped": len(scraped_data),
                "domains": result_data["domains"]
            })
            
            if self._cancelled:
                return self._cancelled_result()
            
            # ================================================================
            # STEP 3: Store Scraped Content
            # ================================================================
            step3 = self._add_step(
                "store_content",
                f"Storing {len(scraped_data)} pages in database"
            )
            self._update_step(step3, AgentStatus.RUNNING)
            self._report_progress("store_content", 2, 6)
            
            self.db.add_scraped_content(project_name, scraped_data)
            
            self._update_step(step3, AgentStatus.COMPLETED, metadata={
                "pages_stored": len(scraped_data)
            })
            
            if self._cancelled:
                return self._cancelled_result()
            
            # ================================================================
            # STEP 4: Generate Prompts
            # ================================================================
            step4 = self._add_step(
                "generate_prompts",
                "Generating relevant questions from content"
            )
            self._update_step(step4, AgentStatus.RUNNING)
            self._report_progress("generate_prompts", 3, 6)
            
            from .promptfile_agent import generate_prompts_from_content
            
            prompt_file_data = generate_prompts_from_content(scraped_data, business_context)
            prompts = prompt_file_data.get("prompts", [])
            
            result_data["prompts_generated"] = len(prompts)
            
            # Store prompt file
            prompt_file_json = json.dumps(prompt_file_data)
            self.db.add_prompt_file(project_name, prompt_file_json, business_context or "")
            
            self._update_step(step4, AgentStatus.COMPLETED, metadata={
                "prompts_generated": len(prompts)
            })
            
            if self._cancelled:
                return self._cancelled_result()
            
            # ================================================================
            # STEP 5: Generate Answers
            # ================================================================
            step5 = self._add_step(
                "generate_answers",
                f"Generating answers for {len(prompts)} prompts"
            )
            self._update_step(step5, AgentStatus.RUNNING)
            self._report_progress("generate_answers", 4, 6)
            
            from .answer_agent_universal import generate_answers_for_business
            
            qa_pairs = generate_answers_for_business(
                prompts,
                scraped_data,
                business_context,
                default_weight
            )
            
            self._update_step(step5, AgentStatus.COMPLETED, metadata={
                "qa_pairs_generated": len(qa_pairs)
            })
            
            if self._cancelled:
                return self._cancelled_result()
            
            # ================================================================
            # STEP 6: Create Embeddings & Store Q&A Pairs
            # ================================================================
            step6 = self._add_step(
                "store_qa_pairs",
                f"Creating embeddings and storing {len(qa_pairs)} Q&A pairs"
            )
            self._update_step(step6, AgentStatus.RUNNING)
            self._report_progress("store_qa_pairs", 5, 6)
            
            from .. import embedding as emb_module
            
            added_count = 0
            errors = []
            
            for i, qa_pair in enumerate(qa_pairs):
                if self._cancelled:
                    return self._cancelled_result()
                
                try:
                    prompt_embedding = emb_module.generate_embedding(
                        qa_pair["prompt"], 
                        embedding_model
                    )
                    self.db.add_qa_pair(
                        project_name,
                        qa_pair["prompt"],
                        qa_pair["response"],
                        prompt_embedding,
                        qa_pair.get("weight", default_weight)
                    )
                    added_count += 1
                    
                except Exception as e:
                    errors.append(f"Q&A {i+1}: {str(e)}")
                    logger.warning(f"Failed to add Q&A pair {i+1}: {e}")
            
            result_data["qa_pairs_added"] = added_count
            
            self._update_step(step6, AgentStatus.COMPLETED, metadata={
                "qa_pairs_added": added_count,
                "errors": len(errors)
            })
            
            self._report_progress("store_qa_pairs", 6, 6)
            
            # ================================================================
            # Return Success Result
            # ================================================================
            return AgentResult(
                success=True,
                message=f"Successfully generated {added_count} Q&A pairs from {len(scraped_data)} pages",
                steps=self.steps,
                data=result_data
            )
            
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            
            # Mark current running step as failed
            for step in self.steps:
                if step.status == AgentStatus.RUNNING:
                    self._update_step(step, AgentStatus.FAILED, error=str(e))
            
            return AgentResult(
                success=False,
                message=f"Agent execution failed: {str(e)}",
                steps=self.steps,
                data=result_data
            )
    
    def _cancelled_result(self) -> AgentResult:
        """Return a cancelled result."""
        for step in self.steps:
            if step.status == AgentStatus.RUNNING:
                self._update_step(step, AgentStatus.CANCELLED)
        
        return AgentResult(
            success=False,
            message="Agent execution was cancelled",
            steps=self.steps
        )


def run_business_website_scraper(
    db_backend,
    project_name: str,
    url: str,
    max_pages: int = 10,
    business_context: Optional[str] = None,
    default_weight: float = 1.0,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> AgentResult:
    """
    Convenience function to run the Business Website Scraper agent.
    
    Args:
        db_backend: Database backend instance
        project_name: Target project name
        url: Website URL to scrape
        max_pages: Maximum pages to scrape
        business_context: Optional business context
        default_weight: Default Q&A weight
        progress_callback: Optional progress callback
        
    Returns:
        AgentResult with execution details
    """
    agent = BusinessWebsiteScraperAgent(
        db_backend=db_backend,
        progress_callback=progress_callback
    )
    
    return agent.run(
        project_name=project_name,
        url=url,
        max_pages=max_pages,
        business_context=business_context,
        default_weight=default_weight
    )
