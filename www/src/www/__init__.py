import requests
import requests.adapters
import time


class _RetryingAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, max_retries: int = 0, backoff0_seconds: float = 0.5, logger=None, sleep=time.sleep):
        super().__init__()
        self.max_retries, self.backoff0_seconds, self._logger, self.sleep = max_retries, backoff0_seconds, logger, sleep

    def send(self, request, **kwargs):
        "part of the HTTPAdapter interface"
        i, backoff_seconds = 0, self.backoff0_seconds
        while True:
            response = super().send(request, **kwargs)
            if (response.status_code // 100 == 2) or (i >= self.max_retries):
                return response
            if self._logger:
                self._logger.warn(
                    "request to %s returned %d %r, waiting %r seconds and retrying",
                    request.url,
                    response.status_code,
                    response.reason,
                    backoff_seconds,
                )
            self.sleep(backoff_seconds)
            i += 1
            backoff_seconds *= 2


class Response(requests.Response):
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


class Session(requests.Session):

    def request(self, *args, **kwargs):
        resp = super().request(*args, **kwargs)
        resp.__class__ = Response
        return resp


def new_session(max_retries: int = 0, backoff0_seconds: float = 0.5, logger=None, sleep=time.sleep) -> requests.Session:
    "create a new session that retries faileds requests up to max_retries times with exponential backoff starting at backoff0_seconds"
    sess = Session()
    adapter = _RetryingAdapter(
        max_retries=max_retries, backoff0_seconds=backoff0_seconds, logger=logger, sleep=sleep
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess
