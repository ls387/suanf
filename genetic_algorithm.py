# -*- coding: utf-8 -*-
"""
遗传算法核心模块
实现智能排课的遗传算法
"""

import random
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import copy

from data_models import *

logger = logging.getLogger(__name__)


class SchedulingGeneticAlgorithm:
    """排课遗传算法"""

    def __init__(self, data: Dict, config: Dict = None):
        self.data = data
        self.config = config or self._default_config()

        # 预处理数据
        self._preprocess_data()

        # 构建查找表
        self._build_lookup_tables()

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "population_size": 100,
            "generations": 200,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "tournament_size": 5,
            "elitism_size": 10,
            "max_stagnation": 50,
            "penalty_scores": {
                "hard_constraint": -99999,
                "teacher_conflict": -10000,
                "class_conflict": -10000,
                "classroom_conflict": -10000,
                "capacity_violation": -5000,
                "blackout_violation": -5000,
                "feature_violation": -5000,
                "thursday_afternoon": -5000,
                "teacher_preference": 100,
                "campus_commute": 200,
                "classroom_continuity": 150,
                "utilization_waste": 1,
                "student_overload": 300,
                "task_relation": 200,
            },
        }

    def _preprocess_data(self):
        """预处理数据"""
        logger.info("开始预处理数据")

        # 按优先级排序教学任务
        self.tasks = self._sort_tasks_by_priority()
        self.task_dict = {task.task_id: task for task in self.tasks}

        # 构建可用教室列表（按容量排序）
        self.classrooms = list(self.data["classrooms"].values())
        self.classrooms.sort(key=lambda x: x.capacity)

        # 构建教师黑名单时间映射
        self.teacher_blackouts = self._build_teacher_blackouts()

        # 构建教师偏好映射
        self.teacher_preferences = self._build_teacher_preferences()

        logger.info(
            f"预处理完成：{len(self.tasks)}个任务，{len(self.classrooms)}个教室"
        )

    def _sort_tasks_by_priority(self) -> List[TeachingTask]:
        """按优先级排序教学任务"""
        tasks = self.data["teaching_tasks"]

        def priority_key(task):
            offering = task.offering
            if not offering:
                return (2, 0, 0)  # 默认优先级

            # 课程性质优先级
            nature_priority = {
                CourseNature.GENERAL: 0,  # 通识最高
                CourseNature.REQUIRED: 1,  # 必修中等
                CourseNature.ELECTIVE: 2,  # 选修最低
            }

            # 学生人数（越多优先级越高）
            student_count = task.student_count or 0

            # 计算该offering的总学时
            offering_tasks = [t for t in tasks if t.offering_id == task.offering_id]
            total_slots = sum(t.slots_count for t in offering_tasks)

            return (
                nature_priority.get(offering.course_nature, 2),
                -student_count,  # 负号表示降序
                -total_slots,  # 负号表示降序
            )

        sorted_tasks = sorted(tasks, key=priority_key)
        logger.info("任务优先级排序完成")
        return sorted_tasks

    def _build_teacher_blackouts(self) -> Dict[str, Set[Tuple[int, int]]]:
        """构建教师黑名单时间映射"""
        blackouts = defaultdict(set)
        for blackout in self.data["teacher_blackout_times"]:
            teacher_id = blackout.teacher_id
            for slot in range(blackout.start_slot, blackout.end_slot + 1):
                blackouts[teacher_id].add((blackout.weekday, slot))
        return dict(blackouts)

    def _build_teacher_preferences(self) -> Dict[str, Dict]:
        """构建教师偏好映射"""
        preferences = defaultdict(lambda: {"preferred": [], "avoided": []})

        for pref in self.data["teacher_preferences"]:
            teacher_id = pref.teacher_id
            if pref.weekday and pref.start_slot and pref.end_slot:
                time_range = (pref.weekday, pref.start_slot, pref.end_slot)
                if pref.preference_type == PreferenceType.PREFERRED:
                    preferences[teacher_id]["preferred"].append(
                        (time_range, pref.penalty_score)
                    )
                else:
                    preferences[teacher_id]["avoided"].append(
                        (time_range, pref.penalty_score)
                    )

        return dict(preferences)

    def _build_lookup_tables(self):
        """构建查找表提高性能"""
        # 按校区分组教室
        self.classrooms_by_campus = defaultdict(list)
        for classroom in self.classrooms:
            self.classrooms_by_campus[classroom.campus_id].append(classroom)

        # 按特征分组教室
        self.classrooms_by_feature = defaultdict(list)
        for classroom in self.classrooms:
            for feature in classroom.features:
                self.classrooms_by_feature[feature].append(classroom)

    def create_individual(self) -> List[Gene]:
        """创建一个个体（染色体）"""
        genes = []

        # 跟踪占用情况
        teacher_schedule = defaultdict(set)  # teacher_id -> {(weekday, slot), ...}
        class_schedule = defaultdict(set)  # class_id -> {(weekday, slot), ...}
        classroom_schedule = defaultdict(set)  # classroom_id -> {(weekday, slot), ...}

        for task in self.tasks:
            gene = self._create_gene_for_task(
                task, teacher_schedule, class_schedule, classroom_schedule
            )
            if gene:
                genes.append(gene)
                self._update_schedules(
                    gene, task, teacher_schedule, class_schedule, classroom_schedule
                )

        return genes

    def _create_gene_for_task(
        self,
        task: TeachingTask,
        teacher_schedule: Dict,
        class_schedule: Dict,
        classroom_schedule: Dict,
    ) -> Optional[Gene]:
        """为单个任务创建基因"""
        if not task.teachers:
            return None

        # 获取有效时间块
        valid_slots = get_valid_time_slots(task.slots_count)

        # 尝试多次找到可行的安排
        max_attempts = 100
        for attempt in range(max_attempts):
            # 随机选择教师
            teacher_id = random.choice(task.teachers)

            # 随机选择时间
            weekday = random.randint(1, 7)
            start_slot, _ = random.choice(valid_slots)

            # 检查周四下午限制
            if weekday == 4 and start_slot >= 6:  # 周四下午
                continue

            # 检查教师黑名单时间
            if self._violates_teacher_blackout(
                teacher_id, weekday, start_slot, task.slots_count
            ):
                continue

            # 检查时间冲突
            if self._has_time_conflict(
                teacher_id,
                task.classes,
                weekday,
                start_slot,
                task.slots_count,
                teacher_schedule,
                class_schedule,
            ):
                continue

            # 选择合适的教室
            classroom = self._select_classroom(
                task, weekday, start_slot, classroom_schedule
            )
            if classroom:
                return Gene(
                    task.task_id,
                    teacher_id,
                    classroom.classroom_id,
                    weekday,
                    start_slot,
                )

        # 如果无法找到可行安排，返回一个随机安排（会在适应度函数中被惩罚）
        teacher_id = random.choice(task.teachers)
        weekday = random.randint(1, 5)  # 避开周末
        start_slot, _ = random.choice(valid_slots)
        classroom = random.choice(self.classrooms)

        return Gene(
            task.task_id, teacher_id, classroom.classroom_id, weekday, start_slot
        )

    def _violates_teacher_blackout(
        self, teacher_id: str, weekday: int, start_slot: int, slots_count: int
    ) -> bool:
        """检查是否违反教师黑名单时间"""
        blackouts = self.teacher_blackouts.get(teacher_id, set())
        for slot in range(start_slot, start_slot + slots_count):
            if (weekday, slot) in blackouts:
                return True
        return False

    def _has_time_conflict(
        self,
        teacher_id: str,
        class_ids: List[str],
        weekday: int,
        start_slot: int,
        slots_count: int,
        teacher_schedule: Dict,
        class_schedule: Dict,
    ) -> bool:
        """检查时间冲突"""
        time_slots = {
            (weekday, slot) for slot in range(start_slot, start_slot + slots_count)
        }

        # 检查教师冲突
        if teacher_schedule[teacher_id] & time_slots:
            return True

        # 检查班级冲突
        for class_id in class_ids:
            if class_schedule[class_id] & time_slots:
                return True

        return False

    def _select_classroom(
        self,
        task: TeachingTask,
        weekday: int,
        start_slot: int,
        classroom_schedule: Dict,
    ) -> Optional[Classroom]:
        """选择合适的教室"""
        time_slots = {
            (weekday, slot) for slot in range(start_slot, start_slot + task.slots_count)
        }

        # 筛选满足容量和特征要求的教室
        suitable_classrooms = []
        for classroom in self.classrooms:
            # 检查容量
            if classroom.capacity < task.student_count:
                continue

            # 检查特征要求
            if not task.required_features.issubset(classroom.features):
                continue

            # 检查时间冲突
            if classroom_schedule[classroom.classroom_id] & time_slots:
                continue

            suitable_classrooms.append(classroom)

        if suitable_classrooms:
            # 优先选择容量接近的教室
            suitable_classrooms.sort(key=lambda x: x.capacity - task.student_count)
            return suitable_classrooms[0]

        return None

    def _update_schedules(
        self,
        gene: Gene,
        task: TeachingTask,
        teacher_schedule: Dict,
        class_schedule: Dict,
        classroom_schedule: Dict,
    ):
        """更新时间占用情况"""
        time_slots = {
            (gene.week_day, slot)
            for slot in range(gene.start_slot, gene.start_slot + task.slots_count)
        }

        teacher_schedule[gene.teacher_id].update(time_slots)
        classroom_schedule[gene.classroom_id].update(time_slots)

        for class_id in task.classes:
            class_schedule[class_id].update(time_slots)

    def fitness(self, individual: List[Gene]) -> float:
        """计算适应度函数"""
        score = 0

        # 构建时间占用表
        teacher_schedule = defaultdict(list)
        class_schedule = defaultdict(list)
        classroom_schedule = defaultdict(list)

        for gene in individual:
            task = self.task_dict[gene.task_id]
            end_slot = gene.start_slot + task.slots_count - 1

            for slot in range(gene.start_slot, end_slot + 1):
                time_key = (gene.week_day, slot)
                teacher_schedule[gene.teacher_id].append(time_key)
                classroom_schedule[gene.classroom_id].append(time_key)

                for class_id in task.classes:
                    class_schedule[class_id].append(time_key)

        # 检查硬约束
        score += self._check_hard_constraints(
            individual, teacher_schedule, class_schedule, classroom_schedule
        )

        # 如果硬约束被违反，直接返回低分
        if score < -50000:
            return score

        # 检查软约束
        score -= self._check_soft_constraints(
            individual, teacher_schedule, class_schedule, classroom_schedule
        )

        return score

    def _check_hard_constraints(
        self,
        individual: List[Gene],
        teacher_schedule: Dict,
        class_schedule: Dict,
        classroom_schedule: Dict,
    ) -> float:
        """检查硬约束"""
        penalty = 0

        # 检查时间冲突
        for schedule_dict, conflict_type in [
            (teacher_schedule, "teacher_conflict"),
            (class_schedule, "class_conflict"),
            (classroom_schedule, "classroom_conflict"),
        ]:
            for entity_id, time_list in schedule_dict.items():
                if len(time_list) != len(set(time_list)):
                    penalty += self.config["penalty_scores"][conflict_type]

        # 检查其他硬约束
        for gene in individual:
            task = self.task_dict[gene.task_id]

            # 检查教室容量
            classroom = self.data["classrooms"][gene.classroom_id]
            if classroom.capacity < task.student_count:
                penalty += self.config["penalty_scores"]["capacity_violation"]

            # 检查特征要求
            if not task.required_features.issubset(classroom.features):
                penalty += self.config["penalty_scores"]["feature_violation"]

            # 检查教师黑名单时间
            if self._violates_teacher_blackout(
                gene.teacher_id, gene.week_day, gene.start_slot, task.slots_count
            ):
                penalty += self.config["penalty_scores"]["blackout_violation"]

            # 检查周四下午限制
            if gene.week_day == 4 and gene.start_slot >= 6:
                penalty += self.config["penalty_scores"]["thursday_afternoon"]

        return penalty

    def _check_soft_constraints(
        self,
        individual: List[Gene],
        teacher_schedule: Dict,
        class_schedule: Dict,
        classroom_schedule: Dict,
    ) -> float:
        """检查软约束"""
        penalty = 0

        # 教师偏好
        penalty += self._check_teacher_preferences(individual)

        # 校区通勤
        penalty += self._check_campus_commute(individual)

        # 连堂课同教室
        penalty += self._check_classroom_continuity(individual)

        # 教室利用率
        penalty += self._check_utilization_waste(individual)

        # 学生负荷
        penalty += self._check_student_overload(class_schedule)

        return penalty

    def _check_teacher_preferences(self, individual: List[Gene]) -> float:
        """检查教师偏好"""
        penalty = 0

        for gene in individual:
            task = self.task_dict[gene.task_id]
            teacher_prefs = self.teacher_preferences.get(gene.teacher_id, {})

            # 检查避免时段
            for (weekday, start_slot, end_slot), penalty_score in teacher_prefs.get(
                "avoided", []
            ):
                if gene.week_day == weekday and not (
                    gene.start_slot + task.slots_count <= start_slot
                    or gene.start_slot >= end_slot + 1
                ):
                    penalty += penalty_score

            # 检查偏好时段
            in_preferred = False
            for (weekday, start_slot, end_slot), penalty_score in teacher_prefs.get(
                "preferred", []
            ):
                if (
                    gene.week_day == weekday
                    and gene.start_slot >= start_slot
                    and gene.start_slot + task.slots_count <= end_slot + 1
                ):
                    in_preferred = True
                    break

            if not in_preferred and teacher_prefs.get("preferred"):
                penalty += self.config["penalty_scores"]["teacher_preference"]

        return penalty

    def _check_campus_commute(self, individual: List[Gene]) -> float:
        """检查校区通勤"""
        penalty = 0

        # 按教师和日期分组
        teacher_daily_campuses = defaultdict(lambda: defaultdict(set))

        for gene in individual:
            classroom = self.data["classrooms"][gene.classroom_id]
            campus_id = classroom.campus_id

            # 判断时段（上午：1-5，下午：6-10，晚上：11-13）
            if gene.start_slot <= 5:
                period = "morning"
            elif gene.start_slot <= 10:
                period = "afternoon"
            else:
                period = "evening"

            teacher_daily_campuses[gene.teacher_id][gene.week_day].add(
                (period, campus_id)
            )

        # 检查每个教师每天的校区数量
        for teacher_id, daily_campuses in teacher_daily_campuses.items():
            for weekday, period_campuses in daily_campuses.items():
                # 提取所有校区
                campuses = set()
                for period, campus_id in period_campuses:
                    campuses.add(campus_id)

                if len(campuses) > 1:
                    penalty += self.config["penalty_scores"]["campus_commute"] * (
                        len(campuses) - 1
                    )

        return penalty

    def _check_classroom_continuity(self, individual: List[Gene]) -> float:
        """检查连堂课同教室"""
        penalty = 0

        # 按教师和日期分组
        teacher_daily_classes = defaultdict(lambda: defaultdict(list))

        for gene in individual:
            task = self.task_dict[gene.task_id]
            teacher_daily_classes[gene.teacher_id][gene.week_day].append(
                (
                    gene.start_slot,
                    gene.start_slot + task.slots_count - 1,
                    gene.classroom_id,
                )
            )

        # 检查连续课程
        for teacher_id, daily_classes in teacher_daily_classes.items():
            for weekday, classes in daily_classes.items():
                classes.sort(key=lambda x: x[0])  # 按开始时间排序

                for i in range(len(classes) - 1):
                    curr_end = classes[i][1]
                    next_start = classes[i + 1][0]
                    curr_classroom = classes[i][2]
                    next_classroom = classes[i + 1][2]

                    # 如果是连续课程但不在同一教室
                    if curr_end + 1 == next_start and curr_classroom != next_classroom:
                        penalty += self.config["penalty_scores"]["classroom_continuity"]

        return penalty

    def _check_utilization_waste(self, individual: List[Gene]) -> float:
        """检查教室利用率"""
        penalty = 0

        for gene in individual:
            task = self.task_dict[gene.task_id]
            classroom = self.data["classrooms"][gene.classroom_id]

            waste = classroom.capacity - task.student_count
            if waste > 0:
                penalty += waste * self.config["penalty_scores"]["utilization_waste"]

        return penalty

    def _check_student_overload(self, class_schedule: Dict) -> float:
        """检查学生负荷"""
        penalty = 0

        for class_id, time_list in class_schedule.items():
            # 按天统计课程数
            daily_count = defaultdict(int)
            for weekday, slot in time_list:
                daily_count[weekday] += 1

            for weekday, count in daily_count.items():
                if count > 8:  # 一天超过8节课
                    penalty += self.config["penalty_scores"]["student_overload"] * (
                        count - 8
                    )

        return penalty

    def crossover(
        self, parent1: List[Gene], parent2: List[Gene]
    ) -> Tuple[List[Gene], List[Gene]]:
        """交叉操作"""
        if random.random() > self.config["crossover_rate"]:
            return parent1[:], parent2[:]

        # 单点交叉
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def mutate(self, individual: List[Gene]) -> List[Gene]:
        """变异操作"""
        mutated = individual[:]

        for i, gene in enumerate(mutated):
            if random.random() < self.config["mutation_rate"]:
                # 随机选择变异类型
                mutation_type = random.choice(["teacher", "time", "classroom"])

                task = self.task_dict[gene.task_id]

                if mutation_type == "teacher" and len(task.teachers) > 1:
                    # 更换教师
                    new_teacher = random.choice(
                        [t for t in task.teachers if t != gene.teacher_id]
                    )
                    mutated[i] = Gene(
                        gene.task_id,
                        new_teacher,
                        gene.classroom_id,
                        gene.week_day,
                        gene.start_slot,
                    )

                elif mutation_type == "time":
                    # 更换时间
                    valid_slots = get_valid_time_slots(task.slots_count)
                    new_weekday = random.randint(1, 7)
                    new_start_slot, _ = random.choice(valid_slots)
                    mutated[i] = Gene(
                        gene.task_id,
                        gene.teacher_id,
                        gene.classroom_id,
                        new_weekday,
                        new_start_slot,
                    )

                elif mutation_type == "classroom":
                    # 更换教室
                    suitable_classrooms = [
                        cr
                        for cr in self.classrooms
                        if (
                            cr.capacity >= task.student_count
                            and task.required_features.issubset(cr.features)
                        )
                    ]
                    if suitable_classrooms:
                        new_classroom = random.choice(suitable_classrooms)
                        mutated[i] = Gene(
                            gene.task_id,
                            gene.teacher_id,
                            new_classroom.classroom_id,
                            gene.week_day,
                            gene.start_slot,
                        )

        return mutated

    def tournament_selection(
        self, population: List[List[Gene]], fitness_scores: List[float]
    ) -> List[Gene]:
        """锦标赛选择"""
        tournament_size = self.config["tournament_size"]
        tournament_indices = random.sample(
            range(len(population)), min(tournament_size, len(population))
        )

        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx][:]

    def evolve(self) -> List[Gene]:
        """进化主循环"""
        logger.info("开始遗传算法进化")

        # 初始化种群
        population = [
            self.create_individual() for _ in range(self.config["population_size"])
        ]

        best_fitness = float("-inf")
        stagnation_count = 0

        for generation in range(self.config["generations"]):
            # 计算适应度
            fitness_scores = [self.fitness(individual) for individual in population]

            # 记录最佳适应度
            current_best = max(fitness_scores)
            if current_best > best_fitness:
                best_fitness = current_best
                stagnation_count = 0
                logger.info(f"第 {generation} 代，新的最佳适应度: {best_fitness:.2f}")
            else:
                stagnation_count += 1

            # 检查停滞
            if stagnation_count >= self.config["max_stagnation"]:
                logger.info(f"算法停滞 {stagnation_count} 代，提前结束")
                break

            # 精英保留
            elite_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True,
            )
            elite_size = self.config["elitism_size"]
            new_population = [population[i][:] for i in elite_indices[:elite_size]]

            # 生成新个体
            while len(new_population) < self.config["population_size"]:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # 截断到目标大小
            population = new_population[: self.config["population_size"]]

            if generation % 20 == 0:
                logger.info(
                    f"第 {generation} 代完成，当前最佳适应度: {current_best:.2f}"
                )

        # 返回最佳个体
        final_fitness_scores = [self.fitness(individual) for individual in population]
        best_idx = max(
            range(len(final_fitness_scores)), key=lambda i: final_fitness_scores[i]
        )

        logger.info(f"进化完成，最终最佳适应度: {final_fitness_scores[best_idx]:.2f}")
        return population[best_idx]
