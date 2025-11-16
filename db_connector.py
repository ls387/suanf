# -*- coding: utf-8 -*-
"""
数据库连接器模块
负责所有数据库交互操作
"""

import pymysql
import logging
from typing import List, Dict, Set, Optional, Tuple
from data_models import *

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """数据库连接器"""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
    ):
        self.connection_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset,
        }
        self.connection = None

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(**self.connection_config)
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """执行查询语句"""
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"查询执行失败: {query}, 错误: {e}")
            raise

    def execute_insert(self, query: str, params: tuple = None) -> int:
        """执行插入语句"""
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"插入执行失败: {query}, 错误: {e}")
            self.connection.rollback()
            raise

    def execute_batch_insert(self, query: str, params_list: List[tuple]):
        """批量插入"""
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(query, params_list)
                self.connection.commit()
                logger.info(f"批量插入 {len(params_list)} 条记录成功")
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            self.connection.rollback()
            raise


class DataLoader:
    """数据加载器"""

    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector

    def load_all_data(self, semester: str) -> Dict:
        """加载所有排课相关数据"""
        logger.info(f"开始加载学期 {semester} 的排课数据")

        data = {
            "campuses": self._load_campuses(),
            "departments": self._load_departments(),
            "majors": self._load_majors(),
            "classes": self._load_classes(),
            "teachers": self._load_teachers(),
            "courses": self._load_courses(),
            "classroom_features": self._load_classroom_features(),
            "classrooms": self._load_classrooms(),
            "course_offerings": self._load_course_offerings(semester),
            "teaching_groups": self._load_teaching_groups(),
            "teaching_tasks": self._load_teaching_tasks(semester),
            "teacher_blackout_times": self._load_teacher_blackout_times(semester),
            "teacher_preferences": self._load_teacher_preferences(),
            "task_relations": self._load_task_relations(semester),
        }

        # 填充教学任务的详细信息
        self._enrich_teaching_tasks(data)

        logger.info("所有数据加载完成")
        return data

    def _load_campuses(self) -> Dict[str, Campus]:
        """加载校区数据"""
        query = "SELECT campus_id, campus_name, address FROM campuses"
        rows = self.db.execute_query(query)
        return {row["campus_id"]: Campus(**row) for row in rows}

    def _load_departments(self) -> Dict[str, Department]:
        """加载院系数据"""
        query = "SELECT department_id, department_name, campus_id FROM departments"
        rows = self.db.execute_query(query)
        return {row["department_id"]: Department(**row) for row in rows}

    def _load_majors(self) -> Dict[str, Major]:
        """加载专业数据"""
        query = "SELECT major_id, major_name, department_id, notes FROM majors"
        rows = self.db.execute_query(query)
        return {row["major_id"]: Major(**row) for row in rows}

    def _load_classes(self) -> Dict[str, Class]:
        """加载班级数据"""
        query = "SELECT class_id, class_name, grade, student_count, major_id, education_system FROM classes"
        rows = self.db.execute_query(query)
        return {row["class_id"]: Class(**row) for row in rows}

    def _load_teachers(self) -> Dict[str, Teacher]:
        """加载教师数据"""
        query = "SELECT teacher_id, teacher_name, department_id, gender, is_external FROM teachers"
        rows = self.db.execute_query(query)
        return {row["teacher_id"]: Teacher(**row) for row in rows}

    def _load_courses(self) -> Dict[str, Course]:
        """加载课程数据"""
        query = (
            "SELECT course_id, course_name, credits, total_hours, notes FROM courses"
        )
        rows = self.db.execute_query(query)
        return {row["course_id"]: Course(**row) for row in rows}

    def _load_classroom_features(self) -> Dict[str, ClassroomFeature]:
        """加载教室特征数据"""
        query = "SELECT feature_id, feature_name, description FROM classroom_features"
        rows = self.db.execute_query(query)
        return {row["feature_id"]: ClassroomFeature(**row) for row in rows}

    def _load_classrooms(self) -> Dict[str, Classroom]:
        """加载教室数据（包含特征）"""
        # 加载基本教室信息
        query = "SELECT classroom_id, classroom_name, building_name, campus_id, classroom_type, capacity, is_available FROM classrooms WHERE is_available = TRUE"
        rows = self.db.execute_query(query)
        classrooms = {}

        for row in rows:
            classroom = Classroom(**row)
            classrooms[row["classroom_id"]] = classroom

        # 加载教室特征关联
        query = """
        SELECT chf.classroom_id, chf.feature_id 
        FROM classroom_has_features chf
        JOIN classrooms c ON chf.classroom_id = c.classroom_id
        WHERE c.is_available = TRUE
        """
        feature_rows = self.db.execute_query(query)

        for row in feature_rows:
            classroom_id = row["classroom_id"]
            if classroom_id in classrooms:
                classrooms[classroom_id].features.add(row["feature_id"])

        return classrooms

    def _load_course_offerings(self, semester: str) -> Dict[int, CourseOffering]:
        """加载开课计划数据"""
        query = "SELECT offering_id, semester, course_id, course_nature, student_count_estimate, start_week, end_week, week_pattern FROM course_offerings WHERE semester = %s"
        rows = self.db.execute_query(query, (semester,))

        offerings = {}
        for row in rows:
            row["course_nature"] = CourseNature(row["course_nature"])
            offerings[row["offering_id"]] = CourseOffering(**row)

        return offerings

    def _load_teaching_groups(self) -> Dict[int, TeachingGroup]:
        """加载教学班数据"""
        query = "SELECT group_id, offering_id, group_name, student_count FROM teaching_groups"
        rows = self.db.execute_query(query)
        return {row["group_id"]: TeachingGroup(**row) for row in rows}

    def _load_teaching_tasks(self, semester: str) -> List[TeachingTask]:
        """加载教学任务数据"""
        query = """
        SELECT tt.task_id, tt.offering_id, tt.group_id, tt.task_sequence, tt.slots_count
        FROM teaching_tasks tt
        JOIN course_offerings co ON tt.offering_id = co.offering_id
        WHERE co.semester = %s
        ORDER BY tt.task_id
        """
        rows = self.db.execute_query(query, (semester,))
        return [TeachingTask(**row) for row in rows]

    def _load_teacher_blackout_times(self, semester: str) -> List[TeacherBlackoutTime]:
        """加载教师禁止时间"""
        query = "SELECT blackout_id, teacher_id, semester, weekday, start_slot, end_slot, reason FROM teacher_blackout_times WHERE semester = %s"
        rows = self.db.execute_query(query, (semester,))
        return [TeacherBlackoutTime(**row) for row in rows]

    def _load_teacher_preferences(self) -> List[TeacherPreference]:
        """加载教师偏好"""
        query = "SELECT preference_id, offering_id, teacher_id, preference_type, weekday, start_slot, end_slot, penalty_score FROM teacher_preferences"
        rows = self.db.execute_query(query)

        preferences = []
        for row in rows:
            row["preference_type"] = PreferenceType(row["preference_type"])
            preferences.append(TeacherPreference(**row))

        return preferences

    def _enrich_teaching_tasks(self, data: Dict):
        """填充教学任务的详细信息"""
        logger.info("开始填充教学任务详细信息")

        # 加载任务对应的教师
        query = """
        SELECT tt.task_id, ot.teacher_id
        FROM teaching_tasks tt
        JOIN offering_teachers ot ON tt.offering_id = ot.offering_id
        JOIN course_offerings co ON tt.offering_id = co.offering_id
        WHERE co.semester = %s
        """
        teacher_rows = self.db.execute_query(
            query, (list(data["course_offerings"].values())[0].semester,)
        )

        task_teachers = {}
        for row in teacher_rows:
            task_id = row["task_id"]
            if task_id not in task_teachers:
                task_teachers[task_id] = []
            task_teachers[task_id].append(row["teacher_id"])

        # 加载任务对应的班级
        query = """
        SELECT tt.task_id, oc.class_id
        FROM teaching_tasks tt
        JOIN offering_classes oc ON tt.offering_id = oc.offering_id
        JOIN course_offerings co ON tt.offering_id = co.offering_id
        WHERE co.semester = %s
        """
        class_rows = self.db.execute_query(
            query, (list(data["course_offerings"].values())[0].semester,)
        )

        task_classes = {}
        for row in class_rows:
            task_id = row["task_id"]
            if task_id not in task_classes:
                task_classes[task_id] = []
            task_classes[task_id].append(row["class_id"])

        # 加载任务对应的设施要求
        query = """
        SELECT tt.task_id, orf.feature_id
        FROM teaching_tasks tt
        JOIN offering_requires_features orf ON tt.offering_id = orf.offering_id
        JOIN course_offerings co ON tt.offering_id = co.offering_id
        WHERE co.semester = %s AND orf.is_mandatory = TRUE
        """
        feature_rows = self.db.execute_query(
            query, (list(data["course_offerings"].values())[0].semester,)
        )

        task_features = {}
        for row in feature_rows:
            task_id = row["task_id"]
            if task_id not in task_features:
                task_features[task_id] = set()
            task_features[task_id].add(row["feature_id"])

        # 填充任务信息
        for task in data["teaching_tasks"]:
            # 填充教师
            task.teachers = task_teachers.get(task.task_id, [])

            # 填充班级
            task.classes = task_classes.get(task.task_id, [])

            # 填充设施要求
            task.required_features = task_features.get(task.task_id, set())

            # 如果没有特殊设施要求，默认需要标准多媒体设备
            if not task.required_features:
                task.required_features.add("MULTIMEDIA_STD")

            # 填充学生人数
            if task.group_id and task.group_id in data["teaching_groups"]:
                task.student_count = (
                    data["teaching_groups"][task.group_id].student_count or 0
                )
            else:
                # 计算所有关联班级的学生总数
                total_students = 0
                for class_id in task.classes:
                    if class_id in data["classes"]:
                        total_students += data["classes"][class_id].student_count or 0
                task.student_count = total_students

            # 填充开课计划信息
            task.offering = data["course_offerings"].get(task.offering_id)

        logger.info("教学任务详细信息填充完成")

    def save_schedule_results(
        self, version_id: int, genes: List[Gene], tasks: Dict[int, TeachingTask]
    ):
        """保存排课结果到数据库"""
        logger.info(f"开始保存排课结果，版本ID: {version_id}")

        # 先删除该版本的现有结果
        delete_query = "DELETE FROM schedules WHERE version_id = %s"
        self.db.execute_query(delete_query, (version_id,))

        # 批量插入新结果
        insert_query = """
        INSERT INTO schedules (version_id, task_id, classroom_id, week_day, start_slot, end_slot)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        params_list = []
        for gene in genes:
            task = tasks[gene.task_id]
            end_slot = gene.start_slot + task.slots_count - 1
            params_list.append(
                (
                    version_id,
                    gene.task_id,
                    gene.classroom_id,
                    gene.week_day,
                    gene.start_slot,
                    end_slot,
                )
            )

        self.db.execute_batch_insert(insert_query, params_list)
        logger.info(f"成功保存 {len(params_list)} 条排课结果")

    def _load_task_relations(self, semester: str) -> List[TaskRelation]:
        """加载任务关系约束"""
        try:
            query = """
            SELECT tr.task_id, tr.related_task_id, tr.relation_type, 
                   tr.time_gap, tr.same_day, tr.description
            FROM task_relations tr
            JOIN teaching_tasks tt ON tr.task_id = tt.task_id
            JOIN course_offerings co ON tt.offering_id = co.offering_id
            WHERE co.semester = %s AND tr.is_active = 1
            """
            rows = self.db.execute_query(query, (semester,))
            return [TaskRelation(**row) for row in rows]
        except Exception as e:
            # 如果表不存在或其他错误，返回空列表
            logger.warning(f"无法加载任务关系约束: {e}")
            return []
