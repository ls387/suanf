# -*- coding: utf-8 -*-
"""
测试数据生成脚本
为排课系统生成测试数据
"""

import pymysql
import random
from datetime import datetime


def connect_db():
    """连接数据库"""
    return pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="paike2",
        charset="utf8mb4",
    )


def insert_test_data():
    """插入测试数据"""
    conn = connect_db()
    cursor = conn.cursor()

    try:
        # 1. 校区数据
        campuses = [
            ("CP001", "主校区", "主校区地址"),
            ("CP002", "南校区", "南校区地址"),
        ]

        cursor.executemany(
            "INSERT INTO campuses (campus_id, campus_name, address) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE campus_name=VALUES(campus_name)",
            campuses,
        )

        # 2. 院系数据
        departments = [
            ("DEPT001", "计算机学院", "CP001"),
            ("DEPT002", "数学学院", "CP001"),
            ("DEPT003", "外语学院", "CP002"),
        ]

        cursor.executemany(
            "INSERT INTO departments (department_id, department_name, campus_id) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE department_name=VALUES(department_name)",
            departments,
        )

        # 3. 专业数据
        majors = [
            ("MAJ001", "计算机科学与技术", "DEPT001"),
            ("MAJ002", "软件工程", "DEPT001"),
            ("MAJ003", "数学与应用数学", "DEPT002"),
            ("MAJ004", "英语", "DEPT003"),
        ]

        cursor.executemany(
            "INSERT INTO majors (major_id, major_name, department_id) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE major_name=VALUES(major_name)",
            majors,
        )

        # 4. 班级数据
        classes = [
            ("CLS001", "计科21-1班", 2021, 45, "MAJ001"),
            ("CLS002", "计科21-2班", 2021, 42, "MAJ001"),
            ("CLS003", "软工21-1班", 2021, 40, "MAJ002"),
            ("CLS004", "数学21-1班", 2021, 38, "MAJ003"),
            ("CLS005", "英语21-1班", 2021, 35, "MAJ004"),
        ]

        cursor.executemany(
            "INSERT INTO classes (class_id, class_name, grade, student_count, major_id) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE class_name=VALUES(class_name)",
            classes,
        )

        # 5. 教师数据
        teachers = [
            ("T001", "张教授", "DEPT001", "男", False),
            ("T002", "李老师", "DEPT001", "女", False),
            ("T003", "王教授", "DEPT002", "男", False),
            ("T004", "刘老师", "DEPT003", "女", False),
            ("T005", "陈教授", "DEPT001", "男", False),
        ]

        cursor.executemany(
            "INSERT INTO teachers (teacher_id, teacher_name, department_id, gender, is_external) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE teacher_name=VALUES(teacher_name)",
            teachers,
        )

        # 6. 课程数据
        courses = [
            ("C001", "数据结构", 3.0, 48),
            ("C002", "数据库原理", 3.0, 48),
            ("C003", "高等数学", 4.0, 64),
            ("C004", "英语听说", 2.0, 32),
            ("C005", "算法设计", 3.0, 48),
        ]

        cursor.executemany(
            "INSERT INTO courses (course_id, course_name, credits, total_hours) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE course_name=VALUES(course_name)",
            courses,
        )

        # 7. 教室特征数据
        features = [
            ("MULTIMEDIA_STD", "标准多媒体设备", "包含投影仪、音响、计算机"),
            ("COMPUTER_LAB", "计算机实验室", "配备计算机的实验室"),
            ("LANGUAGE_LAB", "语音实验室", "用于语言教学的专用教室"),
        ]

        cursor.executemany(
            "INSERT INTO classroom_features (feature_id, feature_name, description) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE feature_name=VALUES(feature_name)",
            features,
        )

        # 8. 教室数据
        classrooms = [
            ("CR001", "A101", "A栋", "CP001", "普通多媒体", 120, True),
            ("CR002", "A102", "A栋", "CP001", "普通多媒体", 100, True),
            ("CR003", "A201", "A栋", "CP001", "计算机机房", 60, True),
            ("CR004", "B101", "B栋", "CP001", "普通多媒体", 80, True),
            ("CR005", "C101", "C栋", "CP002", "语音实验室", 50, True),
            ("CR006", "C102", "C栋", "CP002", "普通多媒体", 90, True),
        ]

        cursor.executemany(
            "INSERT INTO classrooms (classroom_id, classroom_name, building_name, campus_id, classroom_type, capacity, is_available) VALUES (%s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE classroom_name=VALUES(classroom_name)",
            classrooms,
        )

        # 9. 教室特征关联
        classroom_features = [
            ("CR001", "MULTIMEDIA_STD"),
            ("CR002", "MULTIMEDIA_STD"),
            ("CR003", "MULTIMEDIA_STD"),
            ("CR003", "COMPUTER_LAB"),
            ("CR004", "MULTIMEDIA_STD"),
            ("CR005", "MULTIMEDIA_STD"),
            ("CR005", "LANGUAGE_LAB"),
            ("CR006", "MULTIMEDIA_STD"),
        ]

        cursor.executemany(
            "INSERT INTO classroom_has_features (classroom_id, feature_id) VALUES (%s, %s) ON DUPLICATE KEY UPDATE classroom_id=VALUES(classroom_id)",
            classroom_features,
        )

        # 10. 开课计划数据
        offerings = [
            (1, "2025-2026-1", "C001", "必修", 87),  # 计科两个班合班
            (2, "2025-2026-1", "C002", "必修", 87),
            (3, "2025-2026-1", "C003", "必修", 165),  # 计科+软工+数学合班
            (4, "2025-2026-1", "C004", "选修", 35),  # 英语班
            (5, "2025-2026-1", "C005", "必修", 40),  # 软工班
        ]

        cursor.executemany(
            "INSERT INTO course_offerings (offering_id, semester, course_id, course_nature, student_count_estimate) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE semester=VALUES(semester)",
            offerings,
        )

        # 11. 开课计划-班级关联
        offering_classes = [
            (1, "CLS001"),
            (1, "CLS002"),  # 数据结构：计科两班
            (2, "CLS001"),
            (2, "CLS002"),  # 数据库：计科两班
            (3, "CLS001"),
            (3, "CLS002"),
            (3, "CLS003"),
            (3, "CLS004"),  # 高数：三个班
            (4, "CLS005"),  # 英语听说：英语班
            (5, "CLS003"),  # 算法设计：软工班
        ]

        cursor.executemany(
            "INSERT INTO offering_classes (offering_id, class_id) VALUES (%s, %s) ON DUPLICATE KEY UPDATE offering_id=VALUES(offering_id)",
            offering_classes,
        )

        # 12. 开课计划-教师关联
        offering_teachers = [
            (1, "T001", "主讲", None, None),
            (2, "T002", "主讲", None, None),
            (3, "T003", "主讲", None, None),
            (4, "T004", "主讲", None, None),
            (5, "T005", "主讲", None, None),
        ]

        cursor.executemany(
            "INSERT INTO offering_teachers (offering_id, teacher_id, role, start_week, end_week) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE teacher_id=VALUES(teacher_id)",
            offering_teachers,
        )

        # 13. 教学任务数据
        teaching_tasks = [
            (1, 1, None, 1, 3),  # 数据结构，第1次课，3节
            (2, 1, None, 2, 2),  # 数据结构，第2次课，2节
            (3, 2, None, 1, 3),  # 数据库原理，第1次课，3节
            (4, 2, None, 2, 2),  # 数据库原理，第2次课，2节
            (5, 3, None, 1, 3),  # 高等数学，第1次课，3节
            (6, 3, None, 2, 3),  # 高等数学，第2次课，3节
            (7, 4, None, 1, 2),  # 英语听说，第1次课，2节
            (8, 5, None, 1, 3),  # 算法设计，第1次课，3节
            (9, 5, None, 2, 2),  # 算法设计，第2次课，2节
        ]

        cursor.executemany(
            "INSERT INTO teaching_tasks (task_id, offering_id, group_id, task_sequence, slots_count) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE offering_id=VALUES(offering_id)",
            teaching_tasks,
        )

        # 14. 创建排课版本
        cursor.execute(
            "INSERT INTO schedule_versions (version_id, semester, version_name, status, description, created_by) VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE semester=VALUES(semester)",
            (1, "2025-2026-1", "测试版本1", "draft", "用于测试的排课版本", "system"),
        )

        # 15. 教师黑名单时间（示例：T001周一上午不可用）
        cursor.execute(
            "INSERT INTO teacher_blackout_times (teacher_id, semester, weekday, start_slot, end_slot, reason) VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE teacher_id=VALUES(teacher_id)",
            ("T001", "2025-2026-1", 1, 1, 5, "周一上午有会议"),
        )

        # 16. 教师偏好（示例：T002偏好周二下午）
        cursor.execute(
            "INSERT INTO teacher_preferences (offering_id, teacher_id, preference_type, weekday, start_slot, end_slot, penalty_score) VALUES (%s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE teacher_id=VALUES(teacher_id)",
            (2, "T002", "PREFERRED", 2, 6, 10, 100),
        )

        conn.commit()
        print("测试数据插入成功！")

    except Exception as e:
        print(f"插入数据时发生错误: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    insert_test_data()
