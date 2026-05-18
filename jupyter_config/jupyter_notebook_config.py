# Configuration file for lab.

c = get_config()  # noqa


## Supply overrides for the tornado.web.Application that the Jupyter server uses.
#  Default: {}
# This is needed for the FileUpload widget to accept files above 10MB (after base 64
# encoding). This will allow files somewhere over 200MB.
c.ServerApp.tornado_settings = {"websocket_max_message_size": 250 * 1024 * 1024}
