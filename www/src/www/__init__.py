import requests
import requests.adapters
import time


class _retrying_http_adapter(requests.adapters.HTTPAdapter):
    _sleep = time.sleep

    def __init__(self, max_retries: int = 0, backoff0_seconds: float = 0.5):
        super().__init__()
        self.max_retries, self.backoff0_seconds = max_retries, backoff0_seconds

    def send(self, request, **kwargs):
        "part of the HTTPAdapter interface"
        i, backoff_seconds = 0, self.backoff0_seconds
        while True:
            response = super().send(request, **kwargs)
            if (response.status_code // 100 == 2) or (i >= self.max_retries):
                return response
            _retrying_http_adapter._sleep(backoff_seconds)
            i += 1
            backoff_seconds *= 2


class response(requests.Response):
    def raise_for_status(self):
        try:
            super().raise_for_status()
            return
        except requests.HTTPError:
            pass
        raise requests.HTTPError(
            "HTTP error: %d %r for url %s\nResponse body:\n%s"
            % (self.status_code, self.reason, self.url, self.text[:1000])
        )


class session(requests.Session):
    def request(self, *args, **kwargs):
        resp = super().request(*args, **kwargs)
        resp.__class__ = response
        return resp


def new_session(max_retries: int = 0, backoff0_seconds: float = 0.5) -> requests.Session:
    "create a new session that retries faileds requests up to max_retries times with exponential backoff starting at backoff0_seconds"
    sess = session()
    adapter = _retrying_http_adapter(max_retries=max_retries, backoff0_seconds=backoff0_seconds)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess
