WITH ProductFeatures AS (
    -- Primero generamos las características base por producto
    SELECT 
        si.[Stock Item Key],
        si.[Stock Item],
        si.[Size],
        si.[Brand],
        -- Categoría de Producto basada en características del item
        CASE 
            WHEN si.[Size] IN ('L', 'XL', 'XXL') AND si.[Is Chiller Stock] = 1 THEN 'Outerwear'
            WHEN si.[Size] IN ('S', 'M', 'L') AND si.[Is Chiller Stock] = 0 THEN 'Tops'
            WHEN si.[Size] IN ('L', 'XL') THEN 'Bottoms'
            WHEN si.[Size] IN ('XS', 'S') THEN 'Accessories'
            ELSE 'Dresses'
        END as Product_Category,
        -- Subcategoría más específica
        CASE 
            WHEN si.[Is Chiller Stock] = 1 THEN 'Chaquetas'
            WHEN si.[Size] IN ('L', 'XL', 'XXL') THEN 'Abrigos'
            WHEN si.[Size] IN ('S', 'M') THEN 'Camisetas'
            ELSE 'Accesorios'
        END as Subcategory,
        -- Características del material
        CASE 
            WHEN si.[Is Chiller Stock] = 1 THEN 'Técnico'
            WHEN si.[Brand] LIKE '%Natural%' THEN 'Natural'
            ELSE 'Sintético'
        END as Material_Category,
        CASE 
            WHEN si.[Is Chiller Stock] = 1 THEN 'Gore-Tex'
            WHEN si.[Brand] LIKE '%Natural%' THEN 'Algodón'
            ELSE 'Poliéster'
        END as Material_Type,
        -- Características técnicas
        CASE 
            WHEN si.[Is Chiller Stock] = 1 THEN 'Grueso'
            WHEN si.[Size] IN ('XS', 'S') THEN 'Ligero'
            ELSE 'Medio'
        END as Thickness,
        -- Impermeabilidad
        CASE 
            WHEN si.[Is Chiller Stock] = 1 THEN 'Impermeable'
            ELSE 'No'
        END as Waterproof_Rating
),
SalesAnalysis AS (
    -- Análisis de ventas por producto y temporada
    SELECT 
        pf.*,
        d.[Calendar Month Number],
        d.[Calendar Year],
        -- Definir temporada
        CASE 
            WHEN d.[Calendar Month Number] IN (12,1,2) THEN 'Winter'
            WHEN d.[Calendar Month Number] IN (3,4,5) THEN 'Spring'
            WHEN d.[Calendar Month Number] IN (6,7,8) THEN 'Summer'
            ELSE 'Fall'
        END as Season,
        -- Características basadas en ventas
        CASE 
            WHEN SUM(s.[Quantity]) > AVG(s.[Quantity]) THEN 'Muy Alto'
            WHEN s.[Is Cold Room Temperature] = 1 THEN 'Alto'
            WHEN d.[Calendar Month Number] IN (6,7,8) THEN 'Bajo'
            ELSE 'Medio'
        END as Thermal_Rating,
        -- Características de diseño basadas en ventas y temporada
        CASE 
            WHEN d.[Calendar Month Number] IN (6,7,8) THEN 'Vibrante'
            WHEN d.[Calendar Month Number] IN (12,1,2) THEN 'Oscuro'
            WHEN d.[Calendar Month Number] IN (3,4,5) THEN 'Pastel'
            ELSE 'Neutro'
        END as Color_Family,
        CASE 
            WHEN d.[Calendar Month Number] IN (6,7,8) THEN 'Floral'
            WHEN d.[Calendar Month Number] IN (12,1,2) THEN 'Sólido'
            WHEN d.[Calendar Month Number] IN (3,4,5) THEN 'Estampado'
            ELSE 'Rayas'
        END as Pattern,
        CASE 
            WHEN SUM(s.[Quantity]) > AVG(s.[Quantity]) THEN 'Formal'
            WHEN d.[Calendar Month Number] IN (6,7,8) THEN 'Playa'
            WHEN d.[Calendar Month Number] IN (12,1,2) THEN 'Formal'
            ELSE 'Casual'
        END as Style
    FROM [Fact].[Sale] s
    JOIN [Dimension].[Stock Item] si ON s.[Stock Item Key] = si.[Stock Item Key]
    JOIN [Dimension].[Date] d ON s.[Invoice Date Key] = d.[Date]
    JOIN ProductFeatures pf ON si.[Stock Item Key] = pf.[Stock Item Key]
    GROUP BY 
        pf.[Stock Item Key],
        pf.[Stock Item],
        pf.[Size],
        pf.[Brand],
        pf.Product_Category,
        pf.Subcategory,
        pf.Material_Category,
        pf.Material_Type,
        pf.Thickness,
        pf.Waterproof_Rating,
        d.[Calendar Month Number],
        d.[Calendar Year],
        s.[Is Cold Room Temperature]
)
-- Selección final de registros
SELECT 
    Product_Category,
    Subcategory,
    Material_Category,
    Material_Type,
    Thickness,
    Waterproof_Rating,
    Thermal_Rating,
    Color_Family,
    Pattern,
    Style,
    Season
FROM SalesAnalysis
WHERE 
    -- Filtrar combinaciones inválidas
    NOT (Season = 'Summer' AND Thermal_Rating = 'Muy Alto') AND
    NOT (Season = 'Winter' AND Product_Category = 'Swimwear') AND
    NOT (Season = 'Summer' AND Thickness = 'Grueso' AND Product_Category IN ('Tops', 'Dresses'))
ORDER BY 
    NEWID(); -- Aleatorizar resultados para simular variedad
