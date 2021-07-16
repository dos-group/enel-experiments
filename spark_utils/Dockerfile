FROM openjdk:8-jdk-slim

ARG JAR_FILE=target/*dependencies.jar
COPY ${JAR_FILE} app.jar

ENTRYPOINT ["java","-cp","/app.jar", "de.tu_berlin.dos.arm.spark_utils.adjustments.EllisApplicationRunner"]