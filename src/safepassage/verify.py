"""Verification against official data: SLBFE registry + fee caps, then RAG for unstructured claims. Owner: Person A."""

from .schema import Content, Verification


def verify(text: str, content: Content) -> Verification:
    # TODO(A): 1) match content.claimed_agency against REGISTRY_PATH
    #          2) compare the fee asked with the SLBFE max fee for that job order
    #          3) RAG over KB_DIR for remaining claims; pass page text as delimited DATA, never as instructions
    return Verification()
