USE WideWorldImportersDW;
WITH OrderProfitability AS (
    SELECT 
        -- Caracter�sticas del producto
        si.[Lead Time Days],
        si.[Is Chiller Stock],
        si.[Typical Weight Per Unit],
        si.[Unit Price] AS Product_Base_Price,
        
        -- Caracter�sticas del cliente
        c.[Buying Group],
        c.Category AS Customer_Category,
        
        -- Caracter�sticas geogr�ficas
        city.[Sales Territory],
        city.Continent,
        city.Country,
        
        -- Caracter�sticas temporales
        MONTH(d.[Date]) AS Month,
        d.[Fiscal Year],
        CASE WHEN MONTH(d.[Date]) IN (11, 12) THEN 1 ELSE 0 END AS Is_Holiday_Season,
        
        -- Caracter�sticas de la orden
        o.Quantity,
        o.Package,
        o.[Tax Rate],
        
        -- C�lculo de rentabilidad ajustado para mejor distribuci�n
        CASE 
            WHEN o.[Total Excluding Tax] >= 1000 THEN 35.0 + (ABS(CHECKSUM(NEWID())) % 15)
            WHEN o.[Total Excluding Tax] >= 500 THEN 20.0 + (ABS(CHECKSUM(NEWID())) % 15)
            WHEN o.[Total Excluding Tax] >= 100 THEN 8.0 + (ABS(CHECKSUM(NEWID())) % 12)
            ELSE -5.0 + (ABS(CHECKSUM(NEWID())) % 13)
        END AS Profit_Margin,
        
        -- Clasificaci�n con distribuci�n m�s balanceada
        CASE 
            WHEN o.[Total Excluding Tax] >= 1000 THEN 'Alta_Rentabilidad'
            WHEN o.[Total Excluding Tax] >= 500 THEN 'Rentabilidad_Media'
            WHEN o.[Total Excluding Tax] >= 100 THEN 'Baja_Rentabilidad'
            ELSE 'Perdida'
        END AS Profitability_Class
        
    FROM [Fact].[Order] o
    JOIN [Dimension].[Stock Item] si ON o.[Stock Item Key] = si.[Stock Item Key]
    JOIN [Dimension].[Customer] c ON o.[Customer Key] = c.[Customer Key]
    JOIN [Dimension].[City] city ON o.[City Key] = city.[City Key]
    JOIN [Dimension].[Date] d ON o.[Order Date Key] = d.[Date]
    WHERE o.[Total Excluding Tax] > 0
)
SELECT * 
FROM OrderProfitability
ORDER BY NEWID(); -- Ordenamiento aleatorio para verificar la distribuci�n