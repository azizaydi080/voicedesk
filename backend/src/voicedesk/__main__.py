import logging

import uvicorn

from voicedesk.api.app import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, ws_max_size=4 * 1024 * 1024)


if __name__ == "__main__":
    main()
