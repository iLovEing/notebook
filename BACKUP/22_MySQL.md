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
    PRIMARY KEY(`student_id`)
);
```
***修饰词***
```
CREATE TABLE `student` (
    `student_id` INT KEY AUTO_INCREMENT, #自动加一(就可以不用指定了)
    `name` VARCHAR(20) NOT NULL, #不能为空
    `major` VARCHAR(20) UNIQUE，  #不能重复
    `level` VARCHAR(20) DEFAULT, “first"  #默认值
    PRIMARY KEY(`student_id`)
);
```

- **删除表格**
DROP TABLE \`student\`;

- **显示信息**
DESCRIBE \`student\`;

- **添加/删除属性**
ALTER TABLE \`student\` ADD gpa DECIMAL(3, 2);
ALTER TABLE \`student\` DROP COLUMN gpa;


---

### 数据

- **添加数据**
INSERT INTO \`student\` VALUES(1, "小白", "历史");
INSERT INTO \`student\` VALUES(3, "小绿", NULL);
INSERT INTO \`student\`(\`major\`, \`student_id\`) VALUES("力学", 5); //手动指定ID

- **修改数据**
```
SET SQL_SAFE_UPDATES = 0; #关闭安全更新限制

# 把student表中major等于英语的改为英语文学
UPDATE `student`
SET `major` = "英语文学" #注意这两处都没有分号
WHERE `major` = "英语";

# WHERE中也可以用OR，AND关键词表达复杂条件；SET中可以用逗号添加多种操作
```

- **删除数据**
```
DELETE FROM `student`
WHERE `name` = "小黑";
```
