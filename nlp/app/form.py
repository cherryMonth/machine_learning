# coding=utf-8
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, StringField
from wtforms.validators import Length, DataRequired


class Login(FlaskForm):  # 继承自FlaskForm类
    source = TextAreaField('原始文档',
                             validators=[Length(min=1, max=1000, message='内容长度为1~1000'), DataRequired(message='内容不能为空')])
    target = TextAreaField('目标文档',
                             validators=[Length(min=1, max=1000, message='内容长度为1~1000'), DataRequired(message='内容不能为空')])
    source_keyword = StringField()
    target_keyword = StringField()
    score = StringField()
    submit = SubmitField('提交')
