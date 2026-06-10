###############################################################################
# CloudWatch — App Runner application log group + a basic 5xx alarm
#
# App Runner auto-creates /aws/apprunner/<service>/<id>/application and
# /service log groups. We create the application log group explicitly so we can
# set retention (cost control) and reference it. App Runner adopts an existing
# log group of the expected name.
###############################################################################

resource "aws_cloudwatch_log_group" "apprunner_app" {
  name              = "/aws/apprunner/${var.project_name}/application"
  retention_in_days = var.log_retention_days
}

# Optional SNS topic for the alarm (created only if an email is provided).
resource "aws_sns_topic" "alarms" {
  count = var.alarm_email != "" ? 1 : 0
  name  = "${var.project_name}-alarms"
}

resource "aws_sns_topic_subscription" "alarm_email" {
  count     = var.alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# Basic alarm: App Runner 5xx responses. Cheap (first 10 alarms free tier).
resource "aws_cloudwatch_metric_alarm" "apprunner_5xx" {
  alarm_name          = "${var.project_name}-apprunner-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "5xxStatusResponses"
  namespace           = "AWS/AppRunner"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  treat_missing_data  = "notBreaching"
  alarm_description   = "App Runner returned more than 5 5xx responses in 5 minutes."

  dimensions = {
    ServiceName = aws_apprunner_service.backend.service_name
  }

  alarm_actions = var.alarm_email != "" ? [aws_sns_topic.alarms[0].arn] : []
  ok_actions    = var.alarm_email != "" ? [aws_sns_topic.alarms[0].arn] : []
}
