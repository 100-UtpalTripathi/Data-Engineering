
-- Step 1: Create the source and destination tables first with same structures....

-- Source Table
CREATE TABLE SrcTable (
    Id INT PRIMARY KEY,
    Name NVARCHAR(100),
    Description NVARCHAR(255),
    recCreated DATETIME DEFAULT GETDATE()
);

-- Destination Table
CREATE TABLE DesTable (
    Id INT PRIMARY KEY,
    Name NVARCHAR(100),
    Description NVARCHAR(255),
    recCreated DATETIME DEFAULT GETDATE(),
    IsDeleted BIT DEFAULT 0 -- For soft deleting flag
);


-- Step 2 : Insert data into the source Table...

INSERT INTO SrcTable (Id, Name, Description)
VALUES 
(1, 'Name 1', 'Description 1'),
(2, 'Name 2', 'Description 2'),
(3, 'Name 3', 'Description 3');

-- Step 3: Creating the stored procedure for synchronization.....

-- Drop the procedure if it already exists
IF OBJECT_ID('dbo.SyncTables', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SyncTables;
GO

-- Create the procedure
CREATE PROCEDURE SyncTables
AS
BEGIN
    -- Insert new records from SourceTable to TargetTable
    INSERT INTO DesTable (Id, Name, Description, recCreated)
    SELECT s.Id, s.Name, s.Description, s.recCreated
    FROM SrcTable s
    LEFT JOIN DesTable t ON s.Id = t.Id
    WHERE t.Id IS NULL;
    
    -- Update existing records in TargetTable if data has changed in SourceTable
    UPDATE t
    SET t.Name = s.Name,
        t.Description = s.Description,
        t.recCreated = GETDATE()
    FROM DesTable t
    INNER JOIN SrcTable s ON t.Id = s.Id
    WHERE t.Name <> s.Name OR t.Description <> s.Description;

    -- Soft delete records from TargetTable if they are deleted from SourceTable
    UPDATE t
    SET t.IsDeleted = 1
    FROM DesTable t
    LEFT JOIN SrcTable s ON t.Id = s.Id
    WHERE s.Id IS NULL AND t.IsDeleted = 0;
END;
GO


-- Step 4 : Sync both the tables...

SELECT * FROM SrcTable;
SELECT * FROM DesTable;

EXEC SyncTables;

SELECT * FROM SrcTable;
SELECT * FROM DesTable;


-- Step 5 : Testing...

-- Update a record in Source Table
UPDATE SrcTable
SET Name = 'Updated Name 1', Description = 'Updated Description 1'
WHERE Id = 1;

-- Delete a record from Source Table
DELETE FROM SrcTable
WHERE Id = 2;

SELECT * FROM SrcTable

-- Re-run the synchronization procedure
EXEC SyncTables;

-- Check the contents of Source and Destination Table...
SELECT * FROM SrcTable;
SELECT * FROM DesTable;

