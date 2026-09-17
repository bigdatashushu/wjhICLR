"""M10 SandboxReceipt 哈希链单测：篡改检测。"""

from skill3d.sandbox.receipt import GENESIS_HASH, ReceiptChain, verify_chain


def test_chain_valid():
    chain = ReceiptChain("ep-1")
    chain.append("container_start", {"a": 1})
    chain.append("cell_run", {"cell": 0})
    chain.append("container_destroy", {})
    receipts = chain.receipts
    assert len(receipts) == 3
    assert receipts[0].prev_hash == GENESIS_HASH
    assert receipts[1].prev_hash == receipts[0].receipt_hash
    assert verify_chain(receipts)


def test_tamper_payload_detected():
    chain = ReceiptChain("ep-2")
    chain.append("cell_run", {"x": 1})
    r = chain.receipts[0]
    tampered = r.model_copy(update={"event": "container_destroy"})
    assert not verify_chain([tampered])


def test_tamper_order_detected():
    chain = ReceiptChain("ep-3")
    chain.append("a", {})
    chain.append("b", {})
    rs = chain.receipts
    assert not verify_chain([rs[1], rs[0]])


def test_tamper_hash_detected():
    chain = ReceiptChain("ep-4")
    chain.append("a", {"v": 1})
    r = chain.receipts[0]
    forged = r.model_copy(update={"receipt_hash": "f" * 64})
    assert not verify_chain([forged])
