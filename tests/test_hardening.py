import pytest
from pathlib import Path
from src.agents.triage import TriageAgent, KeywordDomainClassifier
from src.agents.extractor import ExtractionRouter
from src.models.schemas import DocumentProfile, OriginType, LayoutComplexity
from src.strategies.fast_text import FastTextExtractor
from src.strategies.vision import VisionExtractor

def test_triage_pluggable_classifier():
    agent = TriageAgent()
    
    class MockClassifier:
        def classify(self, text, rules):
            return ["mock_domain"]
            
    agent.register_classifier(MockClassifier())
    profile = agent.profile("README.md")
    assert "mock_domain" in profile.domain_hints

def test_fast_text_spatial_provenance():
    extractor = FastTextExtractor()
    profile = DocumentProfile(
        document_id="test",
        filename="test.txt",
        origin_type=OriginType.MARKDOWN,
        layout_complexity=LayoutComplexity.SIMPLE,
        page_count=1,
        language="en",
        domain_hints=[],
        estimated_extraction_cost=0.5,
        ocr_required=False
    )
    # Create a dummy file
    Path("test_spatial.txt").write_text("line1\nline2\nline3")
    result = extractor.extract("test_spatial.txt", profile)
    
    assert len(result.document.bounding_boxes) >= 3
    for bbox in result.document.bounding_boxes.values():
        assert bbox.x0 == 0.0
        assert bbox.x1 == 1.0
    
    Path("test_spatial.txt").unlink()

def test_vision_refusal_detection():
    # We can test the internal logic without calling the API
    refusal_text = "I'm sorry, but I cannot read this image as it is too blurry."
    assert VisionExtractor._is_refusal(refusal_text) is True
    
    valid_text = "This is a contract between Party A and Party B."
    assert VisionExtractor._is_refusal(valid_text) is False

def test_router_vision_budget():
    router = ExtractionRouter()
    # Mock a profile with many pages
    big_profile = DocumentProfile(
        document_id="big_doc",
        filename="huge.pdf",
        origin_type=OriginType.SCANNED_PDF, # Usually starts with vision
        layout_complexity=LayoutComplexity.COMPLEX,
        page_count=50, # Exceeds default limit of 10
        language="en",
        domain_hints=[],
        estimated_extraction_cost=500.0,
        ocr_required=True
    )
    
    # Path doesn't need to exist if it breaks early due to budget
    # but we'll use a dummy
    Path("huge.pdf").write_text("dummy")
    
    # It should skip vision and since it's the only one left in the ladder for SCANNED_PDF...
    # wait, initial for scanned is vision. If it skips, it might return empty result.
    doc, ledger = router.route("huge.pdf", big_profile)
    
    assert "Vision escalation skipped" in ledger.warnings[0]
    assert ledger.final_strategy == "none"
    
    Path("huge.pdf").unlink()
