SELECT 
    current.*, 
    three_hours_ago.aws AS "AWS-3h", 
    two_hours_ago.aws AS "AWS-2h", 
    one_hour_ago.aws AS "AWS-1h", 
    one_hour_later.aws AS "AWS+1h"
FROM 
    extracted_weather_data AS current
LEFT JOIN 
    extracted_weather_data AS three_hours_ago
    ON current.row = three_hours_ago.row 
    AND current.col = three_hours_ago.col 
    AND three_hours_ago.date = current.date - INTERVAL '3 hours' 
LEFT JOIN 
    extracted_weather_data AS two_hours_ago 
    ON current.row = two_hours_ago.row 
    AND current.col = two_hours_ago.col 
    AND two_hours_ago.date = current.date - INTERVAL '2 hours' 
LEFT JOIN 
    extracted_weather_data AS one_hour_ago 
    ON current.row = one_hour_ago.row 
    AND current.col = one_hour_ago.col 
    AND one_hour_ago.date = current.date - INTERVAL '1 hour' 
LEFT JOIN 
    extracted_weather_data AS one_hour_later
    ON current.row = one_hour_later.row
    AND current.col = one_hour_later.col
    AND one_hour_later.date = current.date + INTERVAL '1 hours' 
WHERE 
    current.aws IS NOT NULL 
    AND one_hour_ago.aws IS NOT NULL 
    AND two_hours_ago.aws IS NOT NULL 
    AND three_hours_ago.aws IS NOT NULL 
    AND one_hour_later.aws IS NOT NULL 
ORDER BY 
    current.date ASC;