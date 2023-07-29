# [MySQL](https://github.com/iLovEing/notebook/issues/22)

### key
primary key：主键，表格的键值，可以同时设置多个属性，键值唯一
foreign key：外键，可以是自己表格或者其他表格的key（索引用）

同一个属性可以同时是主键&外键

---

## 资料库

- **资料库操作**
CREATE DATABASE \`sql_tutoria\`;  //创建
SHOW DATABASES;  //展示
DROP DATABASE \`sql_tutorial\`;  //删除
USE \`sql_tutoria\`; //使用

- **支持的数据类型**
  - INT                        //整数
  - DECIMAL(m, n)     //小数，m为总位数，n为小数位数
  - VARCHAR(n)         //字符串，n为字符个数
  - BLOB                     //binary large object
  - DATE                     //日期，格式为 YYYY-MM-DD
  - TIMESTAMP          //时间，格式为 YYYY-MM-DD HH:MM:SS

- **创建表格**
```
CREATE TABLE `student` (
    `student_id` INT PRIMARY KEY,
    `name` VARCHAR(20),
    `major` VARCHAR(20)
);
```
***or***
```
CREATE TABLE `student1` (
`student_id` INT,
    `name` VARCHAR(20),
    `major` VARCHAR(20),
    PRIMARY KEY(`student_id`, `name`)  # 支持多个primal key
);
```
***修饰词***
```
CREATE TABLE `student` (
    `student_id` INT AUTO_INCREMENT,  # 自动加一(就可以不用指定了)
    `name` VARCHAR(20) NOT NULL,  # 不能为空
    `major` VARCHAR(20) UNIQUE,  # 不能重复
    `level` VARCHAR(20) DEFAULT “first",  # 默认值
    PRIMARY KEY(`student_id`)
);
```
***带foreign key***
```
CREATE TABLE `student` (
    `student_id` INT KEY AUTO_INCREMENT,
    `name` VARCHAR(20) NOT NULL,
    `major` VARCHAR(20) UNIQUE,
    `brother_id` INT,
    `major_id` INT,
    PRIMARY KEY(`student_id`),
    FOREIGN KEY (`major_id`) REFERENCES `another_table_name`(`stats_name`) ON DELETE SET NULL
);

# 上述写法需要another_table_name已经创建，如果没有，则需要按添加属性写
ALTER TABLE `student`
ADD FOREIGN KEY(`brother_id`)
REFERENCES `student`(`student_id`)  # 可以是自己
ON DELETE SET NULL;
```

- **删除表格**
DROP TABLE \`student\`;

- **显示信息**
DESCRIBE \`student\`;

- **添加/删除属性**
ALTER TABLE \`student\` ADD gpa DECIMAL(3, 2);
ALTER TABLE \`student\` DROP COLUMN gpa;


---

## 修改数据

- **添加数据**
INSERT INTO \`student\` VALUES(1, "小白", "历史");
INSERT INTO \`student\` VALUES(3, "小绿", NULL);
INSERT INTO \`student\`(\`major\`, \`student_id\`) VALUES("力学", 5); //手动指定ID

注意，有foreign key循环依赖的情况，如果另一张表上没有这个key，会创建失败，可以先把属性值设为null

- **修改数据**
```
SET SQL_SAFE_UPDATES = 0;  # 关闭安全更新限制

# 把student表中major等于英语的改为英语文学
UPDATE `student`
SET `major` = "英语文学"  # 注意这两处都没有分号
WHERE `major` = "英语";

# WHERE中也可以用OR，AND关键词表达复杂条件；SET中可以用逗号添加多种操作
```

- **删除数据**
```
DELETE FROM `student`
WHERE `name` = "小黑";
```


---

## 搜索数据
//比较重要，单独列出来

- 取全部数据
```
SELECT *  # 这里*表示所有属性
FROM `student`;
```

- 取部分数据
```
SELECT `name`, `major`  # 取name major两个属性
FROM `student`;
```

- 带排序
```
SELECT *
FROM `student`
ORDER BY `score`, `student_id` DESC;  # 按score，student_id排序，前者优先级高，DESC 表示降序，默认为升序(ASE)
```

- 带限制 
```
SELECT DISTINCT `major`  # DISTINCT 表示去重
FROM `student`
WHERE `major` = `英语` AND `score` <> 70 # <>表示不等于
# WHERE `major` IN("历史"， “生物”)  # in写法，也可以用or表示
LIMIT 100;  # 限制100个人
```

- 聚合函数
```
# count
SELECT COUNT(*) FROM `student`;  # 统计个数
SELECT COUNT(`major`) FROM `student`;  # 统计有major属性的个数
SELECT COUNT(*)
FROM `student`;
WHERE .......;  # 还可以加条件

# AVG
SELECT AVG(`score`) FROM `student`;

# 还有常用的 SUM  MIN MAX
```

- 万能替代
```
#  %代表任意个字符，_代表一个字符
SELECT *
FROM `student`
WHERE `phone` LIKE "%3154";  # 以3154结尾的电话号码
```

- union 聚集
```
# 属性个数要相同， 属性值类型要相同。最后属性会用第一个
SELECT `major`
FROM `student`
UNION
SELECT `name`
FROM `teacher`;

SELECT `major` as `another_name`  # 可以改属性名字
FROM `student`
UNION
SELECT `name`
FROM `teacher`;
```

- join 连接
```
SELECT *  # 选取最终集合的属性（这里是所有）
FROM `teacher`  # teacher表
JOIN `student`  # 添加到student表格里
ON `my_id` = `teacher_id` ; # 索引条件是student的teacher id 等于teacher的my id

SELECT * 
FROM `teacher`
JOIN `student`
ON `teacher`.`my_id` = `student`.`teacher_id` ;  # 详细写法，选择属性也可以这么用

# 使用left join和right jion，表示左边/右边的表全部选中，否则只有匹配条件的才选中
```


- subquery 子查询
```
# 把查询作为条件，这里表示找出大白所有的学生名
SELECT `name`
FROM `student`
WHERE `teacher_id` = (  # 如果子查询有多个结果，这里要加IN
    SELECT `my_id`
    FROM `teacher`
    WHERE `name` = `大白`
);
```
