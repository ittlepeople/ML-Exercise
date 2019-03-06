# coding=utf-8
from OCR import changeScanPDFToPicTest
import cherrypy

cherrypy.config.update({'server.socket_host': '127.27.7.94',
                        'server.socket_port': 8090})

class Service(object):
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def OCR(self):
        inputpdf = cherrypy.request.json['path']
        text = changeScanPDFToPicTest(inputpdf)
        return {'code': 0, 'text': text}


if __name__ == '__main__':
    cherrypy.quickstart((Service()))



