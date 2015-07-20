set @sql = null;
set SESSION group_concat_max_len=10000;
select GROUP_CONCAT(
		DISTINCT(
			CONCAT('max(CASE WHEN uqa.question_id =',id,
				' THEN uqa.answer_id END) AS `ans_for_qid'
                ,id,'` ')
		)
	) INTO @sql
FROM question;

SET @sql = CONCAT("CREATE table user_question_answer_pivot as select uqa.user_id, ", @sql ,
			"   from user_question_answer uqa 
				left join question q 
                on uqa.question_id = q.id 
                group by uqa.user_id");

PREPARE stmt from @sql;
EXECUTE stmt;
deallocate PREPARE STMT;
