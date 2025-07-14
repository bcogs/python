import http.server
import requests
import threading
import traceback
import urllib3.exceptions
import urllib.parse
import unittest
import warnings

try:
    import www
except ModuleNotFoundError:
    import __init__ as www
warnings.simplefilter("ignore", urllib3.exceptions.NotOpenSSLWarning)


class http_handler(http.server.BaseHTTPRequestHandler):
    failures = 0
    failures_id = ""

    def __init__(self, *args, **kwargs):
        self.response_code = 200
        # issue the super().__init__ late, because it calls do_GET
        super().__init__(*args, **kwargs)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(url.query)
        code = self.response_code
        if "failures" in params:
            if http_handler.failures_id != params["failures_id"][0]:
                http_handler.failures, http_handler.failures_id = 0, params["failures_id"][0]
            if http_handler.failures >= int(params["failures"][0]):
                code = 200
            else:
                code = 500
                http_handler.failures += 1
        self.send_response(code, "ok" if code // 100 == 2 else "oh noooo")
        self.end_headers()
        self.wfile.write(b"a cool page that works" if code // 100 == 2 else b"kaputt")

    def log_message(self, format, *args):
        pass


class test_new_session(unittest.TestCase):
    def setUp(self):
        self.server = http.server.HTTPServer(("localhost", 0), http_handler)
        self.url = f"http://localhost:{self.server.server_port}"
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.start()
        self.sleeps = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def test_max_retries(self):
        RETRIES = 3
        sess = www.new_session(max_retries=RETRIES, sleep=self.sleep)
        for failures in range(2 * RETRIES + 1):
            self.sleeps = []
            resp = sess.get(self.url + "?failures=%d&failures_id=test_max_retries%d" % (failures, failures))
            if failures <= RETRIES:
                resp.raise_for_status()
                self.assertIn("cool", resp.text)
            else:
                with self.assertRaises(requests.HTTPError):
                    resp.raise_for_status()
                self.assertIn("kaputt", resp.text)
            self.assertEqual([0.5 * 2**i for i in range(min(failures, 3))], self.sleeps)

    def test_error_details(self):
        sess = www.new_session(max_retries=1, sleep=self.sleep)
        resp = sess.get(self.url + "?failures=100&failures_id=test_error_details")
        try:
            resp.raise_for_status()
            assert not "the above line should have raised an exception"
        except requests.HTTPError:
            tb = traceback.format_exc()
            self.assertIn("500", tb)
            self.assertIn("oh noooo", tb)
            self.assertIn("kaputt", tb)
