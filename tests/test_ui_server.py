"""Unit and HTTP integration tests for the interactive MicroDuck Harness Studio UI server."""

from __future__ import annotations

import threading
import time
import unittest

import httpx

from src.ui_server import EventBroker, HarnessChatSession, LiveDuckEngine, MicroDuckUIHTTPServer, create_app


class UIServerTests(unittest.TestCase):
    def test_multi_turn_session_squat_bow_and_waddle(self) -> None:
        broker = EventBroker()
        events = broker.subscribe()
        engine = LiveDuckEngine(broker=broker, enable_renderer=False, realtime_chunks=False)
        try:
            session = HarnessChatSession(engine=engine, broker=broker, provider="demo")

            # Prompt 1: Hold a squat
            e1 = session.submit_prompt("Lower into a balanced squat and hold.", blocking=True)
            self.assertEqual(e1["status"], "done")
            snap1 = engine.latest_snapshot()
            self.assertFalse(snap1.fallen)
            self.assertLess(snap1.trunk_pos_m[2], 0.100)

            # Prompt 2: Stand back up cleanly from the squat
            e2 = session.submit_prompt("Stand back up.", blocking=True)
            self.assertEqual(e2["status"], "done")
            snap2 = engine.latest_snapshot()
            self.assertFalse(snap2.fallen)
            self.assertGreater(snap2.trunk_pos_m[2], 0.110)

            # Prompt 3: Waddle forward 2 steps via closed-loop <action_chunk> <-> <observation>
            x_before = snap2.trunk_pos_m[0]
            e3 = session.submit_prompt("Waddle forward 2 steps.", blocking=True)
            self.assertEqual(e3["status"], "done")
            snap3 = engine.latest_snapshot()
            self.assertFalse(snap3.fallen)
            self.assertGreater(snap3.trunk_pos_m[0] - x_before, 0.035)
            # Confirm 2 action_chunk steps + 1 done step
            act_steps = [s for s in e3["steps"] if s["kind"] == "action_chunk"]
            self.assertEqual(len(act_steps), 2)
            self.assertIn("<observation", act_steps[0]["observation_xml"])

            # Verify SSE events flowed through the broker
            drained = []
            while not events.empty():
                drained.append(events.get_nowait())
            self.assertTrue(any("chunk_observed" in m for m in drained))
            self.assertTrue(any("prompt_completed" in m for m in drained))
        finally:
            broker.unsubscribe(events)
            engine.close()

    def test_starlette_asgi_app_and_etag_frame(self) -> None:
        import asyncio

        broker = EventBroker()
        engine = LiveDuckEngine(broker=broker, enable_renderer=True, realtime_chunks=False)
        session = HarnessChatSession(engine=engine, broker=broker, provider="demo")
        app = create_app(engine=engine, session=session, broker=broker)

        async def _run() -> None:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                r_html = await client.get("/")
                self.assertEqual(r_html.status_code, 200)
                self.assertIn("MicroDuck Harness", r_html.text)

                r_frame = await client.get("/api/frame.jpg")
                self.assertEqual(r_frame.status_code, 200)
                etag = r_frame.headers.get("etag", "")
                self.assertTrue(etag)

                r_304 = await client.get("/api/frame.jpg", headers={"If-None-Match": etag})
                self.assertIn(r_304.status_code, (200, 304))

                r_chat = await client.post("/api/chat", json={"message": "Look around and nod."})
                self.assertEqual(r_chat.status_code, 200)
                self.assertIn("entry", r_chat.json())
                for _ in range(120):
                    st = (await client.get("/api/state")).json()
                    if not st["busy"] and st["chat_history"]:
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual(st["chat_history"][0]["status"], "done")
                self.assertFalse(st["telemetry"]["fallen"])

        try:
            asyncio.run(_run())
        finally:
            engine.close()

    def test_http_endpoints_and_rendered_jpeg(self) -> None:
        broker = EventBroker()
        engine = LiveDuckEngine(broker=broker, enable_renderer=True, realtime_chunks=False)
        session = HarnessChatSession(engine=engine, broker=broker, provider="demo")
        httpd = MicroDuckUIHTTPServer(("127.0.0.1", 0), engine=engine, session=session, broker=broker)
        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}"

        th = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        th.start()
        try:
            with httpx.Client(timeout=10.0) as client:
                r_html = client.get(f"{base}/")
                self.assertEqual(r_html.status_code, 200)
                self.assertIn("MicroDuck Harness", r_html.text)

                r_state = client.get(f"{base}/api/state")
                self.assertEqual(r_state.status_code, 200)
                st = r_state.json()
                self.assertEqual(st["provider"], "demo")
                self.assertFalse(st["telemetry"]["fallen"])

                # Switch through camera modes including onboard Duck POV and verify valid JPEG render
                for cam_mode in ("studio", "pov", "split"):
                    r_cam = client.post(f"{base}/api/camera", json={"mode": cam_mode, "delta_az": 12.0})
                    self.assertEqual(r_cam.status_code, 200)
                time.sleep(0.12)
                r_jpg = client.get(f"{base}/api/frame.jpg")
                self.assertEqual(r_jpg.status_code, 200)
                self.assertTrue(r_jpg.content.startswith(b"\xff\xd8"))
                self.assertGreater(len(r_jpg.content), 4000)

                # Trigger async chat prompt over HTTP and wait briefly for completion
                r_chat = client.post(f"{base}/api/chat", json={"message": "Perform a deep bow."})
                self.assertEqual(r_chat.status_code, 200)
                for _ in range(40):
                    s_poll = client.get(f"{base}/api/state").json()
                    if not s_poll["busy"] and s_poll["chat_history"]:
                        break
                    time.sleep(0.05)
                self.assertEqual(s_poll["chat_history"][0]["status"], "done")
                self.assertFalse(s_poll["telemetry"]["fallen"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            engine.close()


if __name__ == "__main__":
    unittest.main()
