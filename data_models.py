# -*- coding: utf-8 -*-
"""
数据模型模块
定义排课系统中的核心数据结构
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Dict
from enum import Enum


class CourseNature(Enum):
    """课程性质枚举"""

    GENERAL = "通识"  # 优先级最高
    REQUIRED = "必修"  # 优先级中等
    ELECTIVE = "选修"  # 优先级最低


class PreferenceType(Enum):
    """偏好类型枚举"""

    PREFERRED = "PREFERRED"
    AVOIDED = "AVOIDED"


@dataclass
class Campus:
    """校区"""

    campus_id: str
    campus_name: str
    address: Optional[str] = None


@dataclass
class Department:
    """院系"""

    department_id: str
    department_name: str
    campus_id: Optional[str] = None


@dataclass
class Major:
    """专业"""

    major_id: str
    major_name: Optional[str] = None
    department_id: Optional[str] = None
    education_level: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Class:
    """行政班"""

    class_id: str
    class_name: str
    grade: int
    student_count: Optional[int]
    major_id: str
    # 可选的学制字段，对应数据库中的 education_system 列
    education_system: Optional[str] = None


@dataclass
class Teacher:
    """教师"""

    teacher_id: str
    teacher_name: str
    department_id: str
    gender: Optional[str] = None
    is_external: bool = False


@dataclass
class Course:
    """课程"""

    course_id: str
    course_name: str
    credits: float
    total_hours: int
    # 课程备注，对应数据库 courses.notes，可为空
    notes: Optional[str] = None


@dataclass
class ClassroomFeature:
    """教室特征/设施"""

    feature_id: str
    feature_name: str
    description: Optional[str] = None


@dataclass
class Classroom:
    """教室"""

    classroom_id: str
    classroom_name: Optional[str]
    building_name: Optional[str]
    campus_id: str
    classroom_type: Optional[str]
    capacity: int
    is_available: bool = True
    features: Set[str] = None  # feature_ids

    def __post_init__(self):
        if self.features is None:
            self.features = set()


@dataclass
class CourseOffering:
    """开课计划"""

    offering_id: int
    semester: str
    course_id: str
    course_nature: CourseNature
    student_count_estimate: Optional[int] = None
    # 起始周、结束周和周次模式，对应 course_offerings 表中的列
    start_week: Optional[int] = None
    end_week: Optional[int] = None
    week_pattern: Optional[str] = None


@dataclass
class TeachingGroup:
    """教学班/小组"""

    group_id: int
    offering_id: int
    group_name: str
    student_count: Optional[int]


@dataclass
class TeachingTask:
    """教学任务次"""

    task_id: int
    offering_id: int
    group_id: Optional[int]  # None表示面向整个offering
    task_sequence: int  # 周内第几次课
    slots_count: int  # 需要连排的节数

    # 以下字段在加载时填充
    teachers: List[str] = None  # teacher_ids
    classes: List[str] = None  # class_ids
    student_count: int = 0
    required_features: Set[str] = None  # 必需的feature_ids
    offering: Optional[CourseOffering] = None

    def __post_init__(self):
        if self.teachers is None:
            self.teachers = []
        if self.classes is None:
            self.classes = []
        if self.required_features is None:
            self.required_features = set()


@dataclass
class TeacherBlackoutTime:
    """教师禁止时间"""

    blackout_id: int
    teacher_id: str
    semester: str
    weekday: int  # 1-7
    start_slot: int  # 1-13
    end_slot: int  # 1-13
    reason: Optional[str] = None


@dataclass
class TeacherPreference:
    """教师偏好"""

    preference_id: int
    offering_id: int
    teacher_id: str
    preference_type: PreferenceType
    weekday: Optional[int] = None  # 1-7
    start_slot: Optional[int] = None  # 1-13
    end_slot: Optional[int] = None  # 1-13
    penalty_score: int = 100


@dataclass
class ScheduleVersion:
    """排课方案版本"""

    version_id: int
    semester: str
    version_name: str
    status: str  # draft, published, archived
    description: Optional[str] = None


@dataclass
class Gene:
    """基因 - 代表一个具体的排课安排"""

    task_id: int
    teacher_id: str
    classroom_id: str
    week_day: int  # 1-7
    start_slot: int  # 1-13

    @property
    def end_slot(self) -> int:
        """根据任务的slots_count计算结束节次"""
        # 这里需要在实际使用时获取task信息
        # 暂时返回start_slot，实际使用时需要完善
        return self.start_slot


@dataclass
class Schedule:
    """排课结果"""

    schedule_id: Optional[int]
    version_id: int
    task_id: int
    classroom_id: str
    week_day: int
    start_slot: int
    end_slot: int


# 时间块定义
TIME_WINDOWS = {
    2: [(1, 2), (9, 10)],  # 2节连堂可用时间块
    3: [(3, 5), (6, 8), (11, 13)],  # 3节连堂可用时间块
    4: [(1, 4), (6, 9)],  # 4节连堂可用时间块
}

# 可放置2节课的所有时间块（包括3节块的前2节）
AVAILABLE_SLOTS_2 = [(1, 2), (3, 4), (6, 7), (9, 10), (11, 12)]


def get_valid_time_slots(slots_count: int) -> List[tuple]:
    """
    根据课程节数获取有效的时间块

    Args:
        slots_count: 课程节数

    Returns:
        有效时间块列表 [(start, end), ...]
    """
    if slots_count == 2:
        return AVAILABLE_SLOTS_2
    elif slots_count in TIME_WINDOWS:
        return TIME_WINDOWS[slots_count]
    else:
        raise ValueError(f"不支持的课程节数: {slots_count}")


@dataclass
class TaskRelation:
    """
    任务关系约束（task_relation_constraints）
    relation_type: e.g. 'min_gap_days'|'same_week'|'before'|'after'|'not_same_day'
    min_gap_days: 如果 relation_type == 'min_gap_days'，要求两次任务间隔至少的天数
    penalty: 违反时的惩罚分数（可为软约束）；若为硬约束，可在适应度中设置大惩罚
    """

    relation_id: int
    offering_id: Optional[int]  # 关联的开课计划（若适用）
    task_id_from: int  # 约束起点任务 id
    task_id_to: int  # 约束目标任务 id
    relation_type: str  # 关系类型
    min_gap_days: Optional[int] = None
    same_week: Optional[bool] = None
    penalty: Optional[int] = 0
    notes: Optional[str] = None
