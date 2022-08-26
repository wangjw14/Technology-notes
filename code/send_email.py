#!/bin/env python
#coding=utf-8
import os
import time
import sys
import os
import xlwt
import xlrd
import smtplib
import traceback
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
reload(sys)
sys.setdefaultencoding("utf-8")
execute_day = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
execute_day = execute_day.split(" ")[0]


class SendEmail(object):
    """ SendEmail """
    def __init__(self):
        """ __init__ """
        pass

    def convert_time(self, timestamp):
        """ convert_time """
        time_local = time.localtime(timestamp)
        dt = time.strftime("%Y%m%d-%H%m%s", time_local)
        return dt

    def send(self, src, dst, title, content, files):
        """ send """
        cont = unicode(content, "utf-8", "ignore")
        msg = MIMEMultipart('alternative')
        if dst == "all":
            dst = ['zhangqing17@baidu.com', 'zhengchuanchuan@baidu.com', 'yutianbao@baidu.com', 'wujiangwei@baidu.com', 'qibingjie@baidu.com', 'yunting@baidu.com']
            # dst = ['qibingjie@baidu.com']
        msg['Subject'] = unicode(title, "utf-8", "ignore")
        msg['From'] = src
        msg['To'] = ','.join(dst)
        part = MIMEText(cont, 'html', 'utf-8')
        msg.attach(part)
        print(dst)
        for sent_file in files:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(sent_file, 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(sent_file))
            msg.attach(part)

        s = smtplib.SMTP()
        s.connect('proxy-in.baidu.com')
        s.sendmail(src, dst, msg.as_string())
        s.close()

    def run(self):
        """ run """
        cur_time = time.time()
        cur_date = self.convert_time(cur_time)
        year_month_day_hour = cur_date[:11]
        res_file = './res_all_routine/email_data.xls'
        # 统计条数
        try:
            if not os.path.exists(res_file):
                print "res_file not exists===="
                sys.exit(1)
            html = \
                """
                <html>
                <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
                <title>时下热门-审核及分发结果</title>
                <body>
                <div id="container">
                <div id="content">
                <h3>(注:文件包含两个表：yesterday_shenhe为前一天审核结果，all_using为正在分发全部数据，详细参见附件)</h3>"""

            email_title = '时下热门' + year_month_day_hour + "审核结果及全部在分发数据"
            email_info = html
            filepath = [res_file]
            self.send("zhengchuanchuan@baidu.com", "all", email_title, email_info, filepath)
        except:
            traceback.print_exc()

if __name__ == '__main__':
    obj = SendEmail()
    obj.run()
