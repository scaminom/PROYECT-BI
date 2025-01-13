WITH SalesByPeriod AS (
    SELECT 
        si.[Stock Item Key],
        si.[Size],
        si.[Is Chiller Stock],
        d.[Calendar Month Number],
        d.[Calendar Year],
        d.[Month],
        d.[Fiscal Year],
		c.[State Province],
        SUM(s.[Quantity]) as Total_Quantity_Sold,
        SUM(s.[Total Including Tax]) as Total_Sales_Amount,
        COUNT(DISTINCT s.[Customer Key]) as Unique_Customers,
        SUM(s.[Profit]) as Total_Profit
    FROM [Fact].[Sale] s
    JOIN [Dimension].[Stock Item] si ON s.[Stock Item Key] = si.[Stock Item Key]
    JOIN [Dimension].[Date] d ON s.[Invoice Date Key] = d.[Date]
	JOIN [Dimension].[City] c ON s.[City Key] = c.[City Key]
    GROUP BY 
        si.[Stock Item Key],
        si.[Stock Item],
        si.[Size],
        si.[Is Chiller Stock],
        d.[Calendar Month Number],
        d.[Calendar Year],
        d.[Month],
        d.[Fiscal Year],
		c.[State Province]

)
SELECT 
    *,
    -- Features adicionales para temporalidad
    LAG(Total_Quantity_Sold) OVER(PARTITION BY [Stock Item Key] ORDER BY [Calendar Year], [Calendar Month Number]) as Previous_Month_Sales,
    AVG(Total_Quantity_Sold) OVER(PARTITION BY [Stock Item Key], [Calendar Month Number]) as Avg_Monthly_Sales,
    CASE 
        WHEN [Calendar Month Number] IN (12,1,2) THEN 'Winter'
        WHEN [Calendar Month Number] IN (3,4,5) THEN 'Spring'
        WHEN [Calendar Month Number] IN (6,7,8) THEN 'Summer'
        ELSE 'Fall'
    END as Season
FROM SalesByPeriod
ORDER BY NEWID();
