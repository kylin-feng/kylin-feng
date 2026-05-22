import { preferences } from '@kit.ArkData';
import { fileIo } from '@kit.CoreFileKit';
import fs from '@ohos.file.fs';
import { Context } from '@kit.AbilityKit';
import util from '@ohos.util';

/**
 * 文档导出格式枚举
 */
enum ExportFormat {
  TXT = 'txt',
  MD = 'md',
  DOCX = 'docx'
}

/**
 * 文档状态枚举
 */
enum DocumentStatus {
  DRAFT = 'draft',
  PUBLISHED = 'published',
  ARCHIVED = 'archived'
}

/**
 * 文档接口
 */
export interface Document {
  id: string;
  title: string;
  content: string;
  template: string;
  createTime: string;
  updateTime: string;
  wordCount: number;
  status: DocumentStatus;
  tags: string[];
}

/**
 * 文档统计信息接口
 */
interface DocumentStats {
  wordCount: number;
  charCount: number;
  lineCount: number;
  paragraphCount: number;
}

/**
 * 文档管理服务类
 * 提供文档的保存、加载、导出等功能
 */
export class DocumentService {
  private static instance: DocumentService;
  private context: Context;
  
  private constructor(context: Context) {
    this.context = context;
  }
  
  static getInstance(context: Context): DocumentService {
    if (!DocumentService.instance) {
      DocumentService.instance = new DocumentService(context);
    }
    return DocumentService.instance;
  }

  /**
   * 保存文档
   * @param title 文档标题
   * @param content 文档内容
   * @param template 使用的模板
   * @returns Promise<boolean> 保存是否成功
   */
  async saveDocument(title: string, content: string, template: string = ''): Promise<boolean> {
    try {
      const documentData: Document = {
        id: this.generateDocumentId(),
        title: title || '未命名文档',
        content: content,
        template: template,
        createTime: new Date().toISOString(),
        updateTime: new Date().toISOString(),
        wordCount: this.getWordCount(content),
        status: DocumentStatus.DRAFT,
        tags: []
      };

      const dataStore = await preferences.getPreferences(this.context, 'documents');
      const documents: Document[] = await dataStore.get('documentList', []) as Document[];
      documents.push(documentData);
      await dataStore.put('documentList', documents);
      await dataStore.flush();

      return true;
    } catch (error) {
      console.error('保存文档失败:', error);
      return false;
    }
  }


  /**
   * 获取所有文档列表
   * @returns Promise<Document[]> 文档列表
   */
  async getDocumentList(): Promise<Document[]> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'documents');
      const documents: Document[] = await dataStore.get('documentList', []) as Document[];
      return documents.sort((a: Document, b: Document) => 
        new Date(b.updateTime).getTime() - new Date(a.updateTime).getTime()
      );
    } catch (error) {
      console.error('获取文档列表失败:', error);
      return [];
    }
  }

  /**
   * 加载文档
   * @param documentId 文档ID
   * @returns Promise<Document | null> 文档内容
   */
  async loadDocument(documentId: string): Promise<Document | null> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'documents');
      const documents: Document[] = await dataStore.get('documentList', []) as Document[];
      return documents.find((doc: Document) => doc.id === documentId) || null;
    } catch (error) {
      console.error('加载文档失败:', error);
      return null;
    }
  }

  /**
   * 删除文档
   * @param documentId 文档ID
   * @returns Promise<boolean> 删除是否成功
   */
  async deleteDocument(documentId: string): Promise<boolean> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'documents');
      const documents: Document[] = await dataStore.get('documentList', []) as Document[];
      const updatedDocuments: Document[] = documents.filter((doc: Document) => doc.id !== documentId);
      await dataStore.put('documentList', updatedDocuments);
      await dataStore.flush();
      return true;
    } catch (error) {
      console.error('删除文档失败:', error);
      return false;
    }
  }

  /**
   * 更新文档
   * @param documentId 文档ID
   * @param content 新内容
   * @returns Promise<boolean> 更新是否成功
   */
  async updateDocument(documentId: string, content: string): Promise<boolean> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'documents');
      const documents: Document[] = await dataStore.get('documentList', []) as Document[];
      const documentIndex: number = documents.findIndex((doc: Document) => doc.id === documentId);
      
      if (documentIndex !== -1) {
        documents[documentIndex].content = content;
        documents[documentIndex].updateTime = new Date().toISOString();
        documents[documentIndex].wordCount = this.getWordCount(content);
        
        await dataStore.put('documentList', documents);
        await dataStore.flush();
        return true;
      }
      return false;
    } catch (error) {
      console.error('更新文档失败:', error);
      return false;
    }
  }

  /**
   * 导出文档为文本文件
   * @param documentId 文档ID
   * @param format 导出格式
   * @returns Promise<string> 导出文件路径
   */
  async exportDocument(documentId: string, format: ExportFormat = ExportFormat.TXT): Promise<string> {
    try {
      const document: Document | null = await this.loadDocument(documentId);
      if (!document) {
        throw new Error('文档不存在');
      }

      const fileName: string = `${document.title}.${format}`;
      const filePath: string = `${this.context.filesDir}/exports/${fileName}`;
      
      // 确保导出目录存在
      await this.ensureDirectoryExists(`${this.context.filesDir}/exports`);
      
      let content: string = '';
      switch (format) {
        case ExportFormat.MD:
          content = this.formatAsMarkdown(document);
          break;
        case ExportFormat.DOCX:
          content = this.formatAsDocx(document);
          break;
        case ExportFormat.TXT:
        default:
          content = this.formatAsText(document);
          break;
      }

      // 使用鸿蒙原生文件系统API写入文件
      const file = await fs.open(filePath, fs.OpenMode.WRITE_ONLY | fs.OpenMode.CREATE);
      const encoder = new util.TextEncoder();
      const encodedContent: Uint8Array = encoder.encode(content);
      await fs.write(file.fd, encodedContent);
      await fs.close(file.fd);
      return filePath;
    } catch (error) {
      console.error('导出文档失败:', error);
      if (error instanceof Error) {
        throw new Error(error.message);
      } else {
        throw new Error('导出文档失败');
      }
    }
  }


  /**
   * 获取文档统计信息
   * @param content 文档内容
   * @returns DocumentStats 统计信息
   */
  getDocumentStats(content: string): DocumentStats {
    return {
      wordCount: this.getWordCount(content),
      charCount: content.length,
      lineCount: content.split('\n').length,
      paragraphCount: content.split('\n\n').filter(p => p.trim().length > 0).length
    };
  }

  private generateDocumentId(): string {
    return 'doc_' + String(Date.now()) + '_' + Math.random().toString(36).substr(2, 9);
  }

  private getWordCount(text: string): number {
    // 简单的中英文词数统计
    const chineseWords = text.match(/[\u4e00-\u9fa5]/g) || [];
    const englishWords = text.match(/[a-zA-Z]+/g) || [];
    return Number(chineseWords.length) + Number(englishWords.length);
  }

  private async ensureDirectoryExists(dirPath: string): Promise<void> {
    try {
      await fileIo.mkdir(dirPath, true);
    } catch (error) {
      // 目录可能已存在，忽略错误
    }
  }

  private formatAsText(document: Document): string {
    return `标题: ${document.title}\n` +
           `创建时间: ${document.createTime}\n` +
           `更新时间: ${document.updateTime}\n` +
           `字数: ${document.wordCount}\n` +
           `模板: ${document.template || '无'}\n\n` +
           `内容:\n${document.content}`;
  }

  private formatAsMarkdown(document: Document): string {
    return `# ${document.title}\n\n` +
           `**创建时间:** ${document.createTime}\n` +
           `**更新时间:** ${document.updateTime}\n` +
           `**字数:** ${document.wordCount}\n` +
           `**模板:** ${document.template || '无'}\n\n` +
           `---\n\n` +
           document.content;
  }

  private formatAsDocx(document: Document): string {
    // 简化的DOCX格式，实际应用中需要使用专门的库
    return `<?xml version="1.0" encoding="UTF-8"?>
<document>
  <title>${document.title}</title>
  <metadata>
    <createTime>${document.createTime}</createTime>
    <updateTime>${document.updateTime}</updateTime>
    <wordCount>${document.wordCount}</wordCount>
    <template>${document.template || '无'}</template>
  </metadata>
  <content>${document.content}</content>
</document>`;
  }
}
